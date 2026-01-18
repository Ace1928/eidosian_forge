import logging
from math import fabs
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import Block
from pyomo.core.expr.numvalue import value
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.gdp.plugins.bigm import BigM_Transformation
@TransformationFactory.register('gdp.fix_disjuncts', doc='Fix disjuncts to their current Boolean values and transforms any\n    LogicalConstraints and BooleanVars so that the resulting model is a\n    (MI)(N)LP.')
class GDP_Disjunct_Fixer(Transformation):
    """Fix disjuncts to their current Boolean values.

    This reclassifies all disjuncts in the passed model instance as ctype Block
    and deactivates the constraints and disjunctions within inactive disjuncts.
    In addition, it transforms relvant LogicalConstraints and BooleanVars so
    that the resulting model is a (MI)(N)LP (where it is only mixed-integer
    if the model contains integer-domain Vars or BooleanVars which were not
    indicator_vars of Disjuncs.
    """

    def __init__(self, **kwargs):
        super(GDP_Disjunct_Fixer, self).__init__(**kwargs)
    CONFIG = ConfigDict('gdp.fix_disjuncts')
    CONFIG.declare('GDP_to_MIP_transformation', ConfigValue(default=BigM_Transformation(), domain=_transformation_name_or_object, description="The name of the transformation to call after the 'logical_to_disjunctive' transformation in order to finish transforming to a MI(N)LP.", doc="\n        If there are no logical constraints on the model being transformed,\n        this option is not used. However, if there are logical constraints\n        that involve mixtures of Boolean and integer variables, this option\n        specifies the transformation to use to transform the model with fixed\n        Disjuncts to a MI(N)LP. Uses 'gdp.bigm' by default.\n        "))

    def _apply_to(self, model, **kwds):
        """Fix all disjuncts in the given model and reclassify them to
        Blocks."""
        config = self.config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        self._transformContainer(model)
        for disjunct_object in model.component_objects(Disjunct, descend_into=(Block, Disjunct)):
            disjunct_object.parent_block().reclassify_component_type(disjunct_object, Block)
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(model)
        config.GDP_to_MIP_transformation.apply_to(model)

    def _transformContainer(self, obj):
        """Find all disjuncts in the container and transform them."""
        for disjunct in obj.component_data_objects(ctype=Disjunct, active=True, descend_into=True):
            _bool = disjunct.indicator_var
            if _bool.value is None:
                raise GDP_Error("The value of the indicator_var of Disjunct '%s' is None. All indicator_vars must have values before calling 'fix_disjuncts'." % disjunct.name)
            elif _bool.value:
                disjunct.indicator_var.fix(True)
                self._transformContainer(disjunct)
            else:
                disjunct.deactivate()
        for disjunction in obj.component_data_objects(ctype=Disjunction, active=True, descend_into=True):
            self._transformDisjunctionData(disjunction)

    def _transformDisjunctionData(self, disjunction):
        logical_sum = sum((value(disj.binary_indicator_var) for disj in disjunction.disjuncts))
        if disjunction.xor and (not logical_sum == 1):
            raise GDP_Error('Disjunction %s violated. Expected 1 disjunct to be active, but %s were active.' % (disjunction.name, logical_sum))
        elif not logical_sum >= 1:
            raise GDP_Error('Disjunction %s violated. Expected at least 1 disjunct to be active, but none were active.')
        else:
            disjunction.deactivate()