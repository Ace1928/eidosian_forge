import io
import logging
import sys
from collections.abc import Sequence
from typing import Optional, List, TextIO
from pyomo.common.config import (
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer
class AutoUpdateConfig(ConfigDict):
    """
    This is necessary for persistent solvers.

    Attributes
    ----------
    check_for_new_or_removed_constraints: bool
    check_for_new_or_removed_vars: bool
    check_for_new_or_removed_params: bool
    check_for_new_objective: bool
    update_constraints: bool
    update_vars: bool
    update_parameters: bool
    update_named_expressions: bool
    update_objective: bool
    treat_fixed_vars_as_params: bool
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        if doc is None:
            doc = 'Configuration options to detect changes in model between solves'
        super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.check_for_new_or_removed_constraints: bool = self.declare('check_for_new_or_removed_constraints', ConfigValue(domain=bool, default=True, description='\n                If False, new/old constraints will not be automatically detected on subsequent\n                solves. Use False only when manually updating the solver with opt.add_constraints()\n                and opt.remove_constraints() or when you are certain constraints are not being\n                added to/removed from the model.'))
        self.check_for_new_or_removed_vars: bool = self.declare('check_for_new_or_removed_vars', ConfigValue(domain=bool, default=True, description='\n                If False, new/old variables will not be automatically detected on subsequent \n                solves. Use False only when manually updating the solver with opt.add_variables() and \n                opt.remove_variables() or when you are certain variables are not being added to /\n                removed from the model.'))
        self.check_for_new_or_removed_params: bool = self.declare('check_for_new_or_removed_params', ConfigValue(domain=bool, default=True, description='\n                If False, new/old parameters will not be automatically detected on subsequent \n                solves. Use False only when manually updating the solver with opt.add_parameters() and \n                opt.remove_parameters() or when you are certain parameters are not being added to /\n                removed from the model.'))
        self.check_for_new_objective: bool = self.declare('check_for_new_objective', ConfigValue(domain=bool, default=True, description='\n                If False, new/old objectives will not be automatically detected on subsequent \n                solves. Use False only when manually updating the solver with opt.set_objective() or \n                when you are certain objectives are not being added to / removed from the model.'))
        self.update_constraints: bool = self.declare('update_constraints', ConfigValue(domain=bool, default=True, description='\n                If False, changes to existing constraints will not be automatically detected on \n                subsequent solves. This includes changes to the lower, body, and upper attributes of \n                constraints. Use False only when manually updating the solver with \n                opt.remove_constraints() and opt.add_constraints() or when you are certain constraints \n                are not being modified.'))
        self.update_vars: bool = self.declare('update_vars', ConfigValue(domain=bool, default=True, description='\n                If False, changes to existing variables will not be automatically detected on \n                subsequent solves. This includes changes to the lb, ub, domain, and fixed \n                attributes of variables. Use False only when manually updating the solver with \n                opt.update_variables() or when you are certain variables are not being modified.'))
        self.update_parameters: bool = self.declare('update_parameters', ConfigValue(domain=bool, default=True, description='\n                If False, changes to parameter values will not be automatically detected on \n                subsequent solves. Use False only when manually updating the solver with \n                opt.update_parameters() or when you are certain parameters are not being modified.'))
        self.update_named_expressions: bool = self.declare('update_named_expressions', ConfigValue(domain=bool, default=True, description='\n                If False, changes to Expressions will not be automatically detected on \n                subsequent solves. Use False only when manually updating the solver with \n                opt.remove_constraints() and opt.add_constraints() or when you are certain \n                Expressions are not being modified.'))
        self.update_objective: bool = self.declare('update_objective', ConfigValue(domain=bool, default=True, description='\n                If False, changes to objectives will not be automatically detected on \n                subsequent solves. This includes the expr and sense attributes of objectives. Use \n                False only when manually updating the solver with opt.set_objective() or when you are \n                certain objectives are not being modified.'))
        self.treat_fixed_vars_as_params: bool = self.declare('treat_fixed_vars_as_params', ConfigValue(domain=bool, default=True, visibility=ADVANCED_OPTION, description='\n                This is an advanced option that should only be used in special circumstances. \n                With the default setting of True, fixed variables will be treated like parameters. \n                This means that z == x*y will be linear if x or y is fixed and the constraint \n                can be written to an LP file. If the value of the fixed variable gets changed, we have \n                to completely reprocess all constraints using that variable. If \n                treat_fixed_vars_as_params is False, then constraints will be processed as if fixed \n                variables are not fixed, and the solver will be told the variable is fixed. This means \n                z == x*y could not be written to an LP file even if x and/or y is fixed. However, \n                updating the values of fixed variables is much faster this way.'))