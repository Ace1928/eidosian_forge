import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core import (
from pyomo.core.base import TransformationFactory, Reference
import pyomo.core.expr as EXPR
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import is_child_of, _get_constraint_transBlock, _to_dict
from pyomo.core.util import target_list
from pyomo.network import Port
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref, ReferenceType
@TransformationFactory.register('gdp.bigm', doc='Relax disjunctive model using big-M terms.')
class BigM_Transformation(GDP_to_MIP_Transformation, _BigM_MixIn):
    """Relax disjunctive model using big-M terms.

    Relaxes a disjunctive model into an algebraic model by adding Big-M
    terms to all disjunctive constraints.

    This transformation accepts the following keyword arguments:
        bigM: A user-specified value (or dict) of M values to use (see below)
        targets: the targets to transform [default: the instance]

    M values are determined as follows:
       1) if the constraint appears in the bigM argument dict
       2) if the constraint parent_component appears in the bigM
          argument dict
       3) if any block which is an ancestor to the constraint appears in
          the bigM argument dict
       3) if 'None' is in the bigM argument dict
       4) if the constraint or the constraint parent_component appear in
          a BigM Suffix attached to any parent_block() beginning with the
          constraint's parent_block and moving up to the root model.
       5) if None appears in a BigM Suffix attached to any
          parent_block() between the constraint and the root model.
       6) if the constraint is linear, estimate M using the variable bounds

    M values may be a single value or a 2-tuple specifying the M for the
    lower bound and the upper bound of the constraint body.

    Specifying "bigM=N" is automatically mapped to "bigM={None: N}".

    The transformation will create a new Block with a unique
    name beginning "_pyomo_gdp_bigm_reformulation".  That Block will
    contain an indexed Block named "relaxedDisjuncts", which will hold
    the relaxed disjuncts.  This block is indexed by an integer
    indicating the order in which the disjuncts were relaxed.
    Each block has a dictionary "_constraintMap":

        'srcConstraints': ComponentMap(<transformed constraint>:
                                       <src constraint>)
        'transformedConstraints': ComponentMap(<src constraint>:
                                               <transformed constraint>)

    All transformed Disjuncts will have a pointer to the block their transformed
    constraints are on, and all transformed Disjunctions will have a
    pointer to the corresponding 'Or' or 'ExactlyOne' constraint.

    """
    CONFIG = ConfigDict('gdp.bigm')
    CONFIG.declare('targets', ConfigValue(default=None, domain=target_list, description='target or list of targets that will be relaxed', doc='\n\n        This specifies the list of components to relax. If None (default), the\n        entire model is transformed. Note that if the transformation is done out\n        of place, the list of targets should be attached to the model before it\n        is cloned, and the list will specify the targets on the cloned\n        instance.'))
    CONFIG.declare('bigM', ConfigValue(default=None, domain=_to_dict, description='Big-M value used for constraint relaxation', doc='\n\n        A user-specified value, dict, or ComponentMap of M values that override\n        M-values found through model Suffixes or that would otherwise be\n        calculated using variable domains.'))
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(default=False, domain=bool, description='Boolean indicating whether or not to transform so that the transformed model will still be valid when fixed Vars are unfixed.', doc='\n        This is only relevant when the transformation will be estimating values\n        for M. If True, the transformation will calculate M values assuming that\n        fixed variables will always be fixed to their current values. This means\n        that if a fixed variable is unfixed after transformation, the\n        transformed model is potentially no longer valid. By default, the\n        transformation will assume fixed variables could be unfixed in the\n        future and will use their bounds to calculate the M value rather than\n        their value. Note that this could make for a weaker LP relaxation\n        while the variables remain fixed.\n        '))
    transformation_name = 'bigm'

    def __init__(self):
        super().__init__(logger)
        self._set_up_expr_bound_visitor()

    def _apply_to(self, instance, **kwds):
        self.used_args = ComponentMap()
        with PauseGC():
            try:
                self._apply_to_impl(instance, **kwds)
            finally:
                self._restore_state()
                self.used_args.clear()
                self._expr_bound_visitor.leaf_bounds.clear()
                self._expr_bound_visitor.use_fixed_var_values_as_bounds = False

    def _apply_to_impl(self, instance, **kwds):
        self._process_arguments(instance, **kwds)
        if self._config.assume_fixed_vars_permanent:
            self._expr_bound_visitor.use_fixed_var_values_as_bounds = True
        targets = self._filter_targets(instance)
        self._transform_logical_constraints(instance, targets)
        gdp_tree = self._get_gdp_tree_from_targets(instance, targets)
        preprocessed_targets = gdp_tree.reverse_topological_sort()
        bigM = self._config.bigM
        for t in preprocessed_targets:
            if t.ctype is Disjunction:
                self._transform_disjunctionData(t, t.index(), bigM, parent_disjunct=gdp_tree.parent(t), root_disjunct=gdp_tree.root_disjunct(t))
        _warn_for_unused_bigM_args(bigM, self.used_args, logger)

    def _transform_disjunctionData(self, obj, index, bigM, parent_disjunct=None, root_disjunct=None):
        transBlock, xorConstraint = self._setup_transform_disjunctionData(obj, root_disjunct)
        or_expr = 0
        for disjunct in obj.disjuncts:
            or_expr += disjunct.binary_indicator_var
            self._transform_disjunct(disjunct, bigM, transBlock)
        if obj.xor:
            xorConstraint[index] = or_expr == 1
        else:
            xorConstraint[index] = or_expr >= 1
        obj._algebraic_constraint = weakref_ref(xorConstraint[index])
        obj.deactivate()

    def _transform_disjunct(self, obj, bigM, transBlock):
        if not obj.active:
            return
        suffix_list = _get_bigM_suffix_list(obj)
        arg_list = self._get_bigM_arg_list(bigM, obj)
        relaxationBlock = self._get_disjunct_transformation_block(obj, transBlock)
        relaxationBlock.bigm_src = {}
        self._transform_block_components(obj, obj, bigM, arg_list, suffix_list)
        obj._deactivate_without_fixing_indicator()

    def _transform_constraint(self, obj, disjunct, bigMargs, arg_list, disjunct_suffix_list):
        transBlock = disjunct._transformation_block()
        bigm_src = transBlock.bigm_src
        constraintMap = transBlock._constraintMap
        disjunctionRelaxationBlock = transBlock.parent_block()
        newConstraint = transBlock.transformedConstraints
        for i in sorted(obj.keys()):
            c = obj[i]
            if not c.active:
                continue
            lower = (None, None, None)
            upper = (None, None, None)
            lower, upper = self._get_M_from_args(c, bigMargs, arg_list, lower, upper)
            M = (lower[0], upper[0])
            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' from the BigM argument is %s." % (c.name, str(M)))
            if M[0] is None and c.lower is not None or (M[1] is None and c.upper is not None):
                suffix_list = _get_bigM_suffix_list(c.parent_block(), stopping_block=disjunct)
                suffix_list.extend(disjunct_suffix_list)
                lower, upper = self._update_M_from_suffixes(c, suffix_list, lower, upper)
                M = (lower[0], upper[0])
            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' after checking suffixes is %s." % (c.name, str(M)))
            if c.lower is not None and M[0] is None:
                M = (self._estimate_M(c.body, c)[0] - c.lower, M[1])
                lower = (M[0], None, None)
            if c.upper is not None and M[1] is None:
                M = (M[0], self._estimate_M(c.body, c)[1] - c.upper)
                upper = (M[1], None, None)
            if self._generate_debug_messages:
                logger.debug("GDP(BigM): The value for M for constraint '%s' after estimating (if needed) is %s." % (c.name, str(M)))
            bigm_src[c] = (lower, upper)
            self._add_constraint_expressions(c, i, M, disjunct.binary_indicator_var, newConstraint, constraintMap)
            c.deactivate()

    def _update_M_from_suffixes(self, constraint, suffix_list, lower, upper):
        need_lower = constraint.lower is not None and lower[0] is None
        need_upper = constraint.upper is not None and upper[0] is None
        M = None
        for bigm in suffix_list:
            if constraint in bigm:
                M = bigm[constraint]
                lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, constraint, constraint)
                if not need_lower and (not need_upper):
                    return (lower, upper)
            if constraint.parent_component() in bigm:
                parent = constraint.parent_component()
                M = bigm[parent]
                lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, parent, constraint)
                if not need_lower and (not need_upper):
                    return (lower, upper)
        if M is None:
            for bigm in suffix_list:
                if None in bigm:
                    M = bigm[None]
                    lower, upper, need_lower, need_upper = self._process_M_value(M, lower, upper, need_lower, need_upper, bigm, None, constraint)
                if not need_lower and (not need_upper):
                    return (lower, upper)
        return (lower, upper)

    @deprecated('The get_m_value_src function is deprecated. Use the get_M_value_src function if you need source information or the get_M_value function if you only need values.', version='5.7.1')
    def get_m_value_src(self, constraint):
        transBlock = _get_constraint_transBlock(constraint)
        (lower_val, lower_source, lower_key), (upper_val, upper_source, upper_key) = transBlock.bigm_src[constraint]
        if constraint.lower is not None and constraint.upper is not None and (not lower_source is upper_source or not lower_key is upper_key):
            raise GDP_Error('This is why this method is deprecated: The lower and upper M values for constraint %s came from different sources, please use the get_M_value_src method.' % constraint.name)
        if constraint.lower is not None and lower_source is not None:
            return (lower_source, lower_key)
        if constraint.upper is not None and upper_source is not None:
            return (upper_source, upper_key)
        return (lower_val, upper_val)

    def get_M_value_src(self, constraint):
        """Return a tuple indicating how the M value used to transform
        constraint was specified. (In particular, this can be used to
        verify which BigM Suffixes were actually necessary to the
        transformation.)

        Return is of the form: ((lower_M_val, lower_M_source, lower_M_key),
                                (upper_M_val, upper_M_source, upper_M_key))

        If the constraint does not have a lower bound (or an upper bound),
        the first (second) element will be (None, None, None). Note that if
        a constraint is of the form a <= expr <= b or is an equality constraint,
        it is not necessarily true that the source of lower_M and upper_M
        are the same.

        If the M value came from an arg, source is the  dictionary itself and
        key is the key in that dictionary which gave us the M value.

        If the M value came from a Suffix, source is the BigM suffix used and
        key is the key in that Suffix.

        If the transformation calculated the value, both source and key are
        None.

        Parameters
        ----------
        constraint: Constraint, which must be in the subtree of a transformed
                    Disjunct
        """
        transBlock = _get_constraint_transBlock(constraint)
        return transBlock.bigm_src[constraint]

    def get_M_value(self, constraint):
        """Returns the M values used to transform constraint. Return is a tuple:
        (lower_M_value, upper_M_value). Either can be None if constraint does
        not have a lower or upper bound, respectively.

        Parameters
        ----------
        constraint: Constraint, which must be in the subtree of a transformed
                    Disjunct
        """
        transBlock = _get_constraint_transBlock(constraint)
        lower, upper = transBlock.bigm_src[constraint]
        return (lower[0], upper[0])

    def get_all_M_values_by_constraint(self, model):
        """Returns a dictionary mapping each constraint to a tuple:
        (lower_M_value, upper_M_value), where either can be None if the
        constraint does not have a lower or upper bound (respectively).

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
        m_values = {}
        for disj in model.component_data_objects(Disjunct, active=None, descend_into=(Block, Disjunct)):
            transBlock = disj.transformation_block
            if transBlock is not None:
                if hasattr(transBlock, 'bigm_src'):
                    for cons in transBlock.bigm_src:
                        m_values[cons] = self.get_M_value(cons)
        return m_values

    def get_largest_M_value(self, model):
        """Returns the largest M value for any constraint on the model.

        Parameters
        ----------
        model: A GDP model that has been transformed with BigM
        """
        return max((max((abs(m) for m in m_values if m is not None)) for m_values in self.get_all_M_values_by_constraint(model).values()))