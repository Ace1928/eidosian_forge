import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def _transformBlock(self, block, currentds):
    self._fe = {}
    for ds in block.component_objects(ContinuousSet, descend_into=True):
        if currentds is None or currentds == ds.name:
            if 'scheme' in ds.get_discretization_info():
                raise DAE_Error("Attempting to discretize ContinuousSet '%s' after it has already been discretized. " % ds.name)
            generate_finite_elements(ds, self._nfe[currentds])
            if not ds.get_changed():
                if len(ds) - 1 > self._nfe[currentds]:
                    logger.warning("More finite elements were found in ContinuousSet '%s' than the number of finite elements specified in apply. The larger number of finite elements will be used." % ds.name)
            self._nfe[ds.name] = len(ds) - 1
            self._fe[ds.name] = list(ds)
            generate_colloc_points(ds, self._tau[currentds])
            disc_info = ds.get_discretization_info()
            disc_info['nfe'] = self._nfe[ds.name]
            disc_info['ncp'] = self._ncp[currentds]
            disc_info['tau_points'] = self._tau[currentds]
            disc_info['adot'] = self._adot[currentds]
            disc_info['adotdot'] = self._adotdot[currentds]
            disc_info['afinal'] = self._afinal[currentds]
            disc_info['scheme'] = self._scheme_name
    expand_components(block)
    for d in block.component_objects(DerivativeVar, descend_into=True):
        dsets = d.get_continuousset_list()
        for i in ComponentSet(dsets):
            if currentds is None or i.name == currentds:
                oldexpr = d.get_derivative_expression()
                loc = d.get_state_var()._contset[i]
                count = dsets.count(i)
                if count >= 3:
                    raise DAE_Error("Error discretizing '%s' with respect to '%s'. Current implementation only allows for taking the first or second derivative with respect to a particular ContinuousSet" % (d.name, i.name))
                scheme = self._scheme[count - 1]
                newexpr = create_partial_expression(scheme, oldexpr, i, loc)
                d.set_derivative_expression(newexpr)
                if self._scheme_name == 'LAGRANGE-LEGENDRE':
                    add_continuity_equations(d.parent_block(), d, i, loc)
        if d.is_fully_discretized():
            add_discretization_equations(d.parent_block(), d)
            d.parent_block().reclassify_component_type(d, Var)
            reclassified_list = getattr(block, '_pyomo_dae_reclassified_derivativevars', None)
            if reclassified_list is None:
                block._pyomo_dae_reclassified_derivativevars = list()
                reclassified_list = block._pyomo_dae_reclassified_derivativevars
            reclassified_list.append(d)
    if block_fully_discretized(block):
        if block.contains_component(Integral):
            for i in block.component_objects(Integral, descend_into=True):
                i.parent_block().reclassify_component_type(i, Expression)
                i.clear()
                i._constructed = False
                i.construct()
            for k in block.component_objects(Objective, descend_into=True):
                k.clear()
                k._constructed = False
                k.construct()