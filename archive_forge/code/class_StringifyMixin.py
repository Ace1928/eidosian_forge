import base64
import inspect
import builtins
class StringifyMixin(object):
    _TYPE = {}
    "_TYPE class attribute is used to annotate types of attributes.\n\n    This type information is used to find an appropriate conversion for\n    a JSON style dictionary.\n\n    Currently the following types are implemented.\n\n    ========= =============\n    Type      Description\n    ========= =============\n    ascii     US-ASCII\n    utf-8     UTF-8\n    asciilist list of ascii\n    ========= =============\n\n    Example::\n        _TYPE = {\n            'ascii': [\n                'hw_addr',\n            ],\n            'utf-8': [\n                'name',\n            ]\n        }\n    "
    _class_prefixes = []
    _class_suffixes = []
    _base_attributes = []
    _opt_attributes = []

    def stringify_attrs(self):
        """an override point for sub classes"""
        return obj_python_attrs(self)

    def __str__(self):
        return self.__class__.__name__ + '(' + ','.join(('%s=%s' % (k, repr(v)) for k, v in self.stringify_attrs())) + ')'
    __repr__ = __str__

    @classmethod
    def _is_class(cls, dict_):
        assert isinstance(dict_, dict)
        if len(dict_) != 1:
            return False
        k = list(dict_.keys())[0]
        if not isinstance(k, (bytes, str)):
            return False
        for p in cls._class_prefixes:
            if k.startswith(p):
                return True
        for p in cls._class_suffixes:
            if k.endswith(p):
                return True
        return False

    @classmethod
    def _get_type(cls, k):
        if hasattr(cls, '_TYPE'):
            for t, attrs in cls._TYPE.items():
                if k in attrs:
                    return _types[t]
        return None

    @classmethod
    def _get_encoder(cls, k, encode_string):
        t = cls._get_type(k)
        if t:
            return t.encode
        return cls._get_default_encoder(encode_string)

    @classmethod
    def _encode_value(cls, k, v, encode_string=base64.b64encode):
        return cls._get_encoder(k, encode_string)(v)

    @classmethod
    def _get_default_encoder(cls, encode_string):

        def _encode(v):
            if isinstance(v, (bytes, str)):
                if isinstance(v, str):
                    v = v.encode('utf-8')
                json_value = encode_string(v)
                json_value = json_value.decode('ascii')
            elif isinstance(v, list):
                json_value = [_encode(ve) for ve in v]
            elif isinstance(v, dict):
                json_value = _mapdict(_encode, v)
                json_value = _mapdict_key(str, json_value)
                assert not cls._is_class(json_value)
            else:
                try:
                    json_value = v.to_jsondict()
                except Exception:
                    json_value = v
            return json_value
        return _encode

    def to_jsondict(self, encode_string=base64.b64encode):
        """
        This method returns a JSON style dict to describe this object.

        The returned dict is compatible with json.dumps() and json.loads().

        Suppose ClassName object inherits StringifyMixin.
        For an object like the following::

            ClassName(Param1=100, Param2=200)

        this method would produce::

            { "ClassName": {"Param1": 100, "Param2": 200} }

        This method takes the following arguments.

        .. tabularcolumns:: |l|L|

        =============  =====================================================
        Argument       Description
        =============  =====================================================
        encode_string  (Optional) specify how to encode attributes which has
                       python 'str' type.
                       The default is base64.
                       This argument is used only for attributes which don't
                       have explicit type annotations in _TYPE class attribute.
        =============  =====================================================
        """
        dict_ = {}
        encode = lambda key, val: self._encode_value(key, val, encode_string)
        for k, v in obj_attrs(self):
            dict_[k] = encode(k, v)
        return {self.__class__.__name__: dict_}

    @classmethod
    def cls_from_jsondict_key(cls, k):
        import sys
        mod = sys.modules[cls.__module__]
        return getattr(mod, k)

    @classmethod
    def obj_from_jsondict(cls, jsondict, **additional_args):
        assert len(jsondict) == 1
        for k, v in jsondict.items():
            obj_cls = cls.cls_from_jsondict_key(k)
            return obj_cls.from_jsondict(v, **additional_args)

    @classmethod
    def _get_decoder(cls, k, decode_string):
        t = cls._get_type(k)
        if t:
            return t.decode
        return cls._get_default_decoder(decode_string)

    @classmethod
    def _decode_value(cls, k, json_value, decode_string=base64.b64decode, **additional_args):
        return cls._get_decoder(k, decode_string)(json_value)

    @classmethod
    def _get_default_decoder(cls, decode_string):

        def _decode(json_value, **additional_args):
            if isinstance(json_value, (bytes, str)):
                v = decode_string(json_value)
            elif isinstance(json_value, list):
                v = [_decode(jv) for jv in json_value]
            elif isinstance(json_value, dict):
                if cls._is_class(json_value):
                    v = cls.obj_from_jsondict(json_value, **additional_args)
                else:
                    v = _mapdict(_decode, json_value)
                    try:
                        v = _mapdict_key(int, v)
                    except ValueError:
                        pass
            else:
                v = json_value
            return v
        return _decode

    @staticmethod
    def _restore_args(dict_):

        def restore(k):
            if k in _RESERVED_KEYWORD:
                return k + '_'
            return k
        return _mapdict_key(restore, dict_)

    @classmethod
    def from_jsondict(cls, dict_, decode_string=base64.b64decode, **additional_args):
        """Create an instance from a JSON style dict.

        Instantiate this class with parameters specified by the dict.

        This method takes the following arguments.

        .. tabularcolumns:: |l|L|

        =============== =====================================================
        Argument        Descrpition
        =============== =====================================================
        dict\\_          A dictionary which describes the parameters.
                        For example, {"Param1": 100, "Param2": 200}
        decode_string   (Optional) specify how to decode strings.
                        The default is base64.
                        This argument is used only for attributes which don't
                        have explicit type annotations in _TYPE class
                        attribute.
        additional_args (Optional) Additional kwargs for constructor.
        =============== =====================================================
        """
        decode = lambda k, x: cls._decode_value(k, x, decode_string, **additional_args)
        kwargs = cls._restore_args(_mapdict_kv(decode, dict_))
        try:
            return cls(**dict(kwargs, **additional_args))
        except TypeError:
            print('CLS %s' % cls)
            print('ARG %s' % dict_)
            print('KWARG %s' % kwargs)
            raise

    @classmethod
    def set_classes(cls, registered_dict):
        cls._class_prefixes.extend([v.__name__ for v in registered_dict.values()])