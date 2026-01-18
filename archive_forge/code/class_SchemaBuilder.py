from . import validators
from . import schema
from . import compound
from . import htmlfill
class SchemaBuilder:

    def __init__(self, validators=default_validators):
        self.validators = validators
        self._schema = None

    def reset(self):
        self._schema = schema.Schema()

    def schema(self):
        return self._schema

    def listen_input(self, parser, tag, attrs):
        get_attr = parser.get_attr
        name = get_attr(attrs, 'name')
        if not name:
            return
        v = compound.All(validators.Identity())
        add_to_end = None
        if tag.lower() == 'input':
            type_attr = get_attr(attrs, 'type').lower().strip()
            if type_attr == 'submit':
                v.validators.append(validators.Bool())
            elif type_attr == 'checkbox':
                v.validators.append(validators.Wrapper(to_python=force_list))
            elif type_attr == 'file':
                add_to_end = validators.FieldStorageUploadConverter()
        message = get_attr(attrs, 'form:message')
        required = to_bool(get_attr(attrs, 'form:required', 'false'))
        if required:
            v.validators.append(validators.NotEmpty(messages=get_messages(validators.NotEmpty, message)))
        else:
            v.validators[0].if_missing = False
        if add_to_end:
            v.validators.append(add_to_end)
        v_type = get_attr(attrs, 'form:validate', None)
        if v_type:
            pos = v_type.find(':')
            if pos != -1:
                args = (v_type[pos + 1:],)
                v_type = v_type[:pos]
            else:
                args = ()
            v_type = v_type.lower()
            v_class = self.validators.get(v_type)
            if not v_class:
                raise ValueError('Invalid validation type: %r' % v_type)
            kw_args = {'messages': get_messages(v_class, message)}
            v_inst = v_class(*args, **kw_args)
            v.validators.append(v_inst)
        self._schema.add_field(name, v)