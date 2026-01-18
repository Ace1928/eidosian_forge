from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
class HOTParamSchema(parameters.Schema):
    """HOT parameter schema."""
    KEYS = TYPE, DESCRIPTION, DEFAULT, SCHEMA, CONSTRAINTS, HIDDEN, LABEL, IMMUTABLE = ('type', 'description', 'default', 'schema', 'constraints', 'hidden', 'label', 'immutable')
    TYPES = STRING, NUMBER, LIST, MAP, BOOLEAN = ('string', 'number', 'comma_delimited_list', 'json', 'boolean')
    PARAMETER_KEYS = KEYS

    @classmethod
    def _constraint_from_def(cls, constraint):
        desc = constraint.get(DESCRIPTION)
        if RANGE in constraint:
            cdef = constraint.get(RANGE)
            cls._check_dict(cdef, RANGE_KEYS, 'range constraint')
            return constr.Range(parameters.Schema.get_num(MIN, cdef), parameters.Schema.get_num(MAX, cdef), desc)
        elif LENGTH in constraint:
            cdef = constraint.get(LENGTH)
            cls._check_dict(cdef, RANGE_KEYS, 'length constraint')
            return constr.Length(parameters.Schema.get_num(MIN, cdef), parameters.Schema.get_num(MAX, cdef), desc)
        elif ALLOWED_VALUES in constraint:
            cdef = constraint.get(ALLOWED_VALUES)
            return constr.AllowedValues(cdef, desc)
        elif ALLOWED_PATTERN in constraint:
            cdef = constraint.get(ALLOWED_PATTERN)
            return constr.AllowedPattern(cdef, desc)
        elif CUSTOM_CONSTRAINT in constraint:
            cdef = constraint.get(CUSTOM_CONSTRAINT)
            return constr.CustomConstraint(cdef, desc)
        else:
            raise exception.InvalidSchemaError(message=_('No constraint expressed'))

    @classmethod
    def _constraints(cls, param_name, schema_dict):
        constraints = schema_dict.get(cls.CONSTRAINTS)
        if constraints is None:
            return
        if not isinstance(constraints, list):
            raise exception.InvalidSchemaError(message=_('Invalid parameter constraints for parameter %s, expected a list') % param_name)
        for constraint in constraints:
            cls._check_dict(constraint, PARAM_CONSTRAINTS, 'parameter constraints')
            yield cls._constraint_from_def(constraint)

    @classmethod
    def from_dict(cls, param_name, schema_dict):
        """Return a Parameter Schema object from a legacy schema dictionary.

        :param param_name: name of the parameter owning the schema; used
               for more verbose logging
        :type  param_name: str
        """
        cls._validate_dict(param_name, schema_dict)
        return cls(schema_dict[cls.TYPE], description=schema_dict.get(HOTParamSchema.DESCRIPTION), default=schema_dict.get(HOTParamSchema.DEFAULT), constraints=list(cls._constraints(param_name, schema_dict)), hidden=schema_dict.get(HOTParamSchema.HIDDEN, False), label=schema_dict.get(HOTParamSchema.LABEL), immutable=schema_dict.get(HOTParamSchema.IMMUTABLE, False))