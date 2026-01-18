import enum
import typing
class _SpnegoErrorRegistry(type):
    __registry: typing.Dict[int, typing.Type] = {}
    __gssapi_map: typing.Dict[int, int] = {}
    __sspi_map: typing.Dict[int, int] = {}

    def __init__(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        error_code = getattr(cls, 'ERROR_CODE', None)
        if error_code is not None and error_code not in cls.__registry:
            cls.__registry[error_code] = cls
        for system_attr, mapping in [('_GSSAPI_CODE', cls.__gssapi_map), ('_SSPI_CODE', cls.__sspi_map)]:
            codes = getattr(cls, system_attr, None)
            if codes is None:
                continue
            if not isinstance(codes, (list, tuple)):
                codes = [codes]
            for c in codes:
                mapping[c] = error_code or 0

    def __call__(cls, error_code: typing.Optional[int]=None, base_error: typing.Optional[Exception]=None, *args: typing.Any, **kwargs: typing.Any) -> '_SpnegoErrorRegistry':
        error_code = error_code if error_code is not None else getattr(cls, 'ERROR_CODE', None)
        if error_code is None:
            if not base_error:
                raise ValueError('%s requires either an error_code or base_error' % cls.__name__)
            maj_code = getattr(base_error, 'maj_code', None)
            winerror = getattr(base_error, 'winerror', None)
            if maj_code is not None:
                error_code = cls.__gssapi_map.get(maj_code, None)
            elif winerror is not None:
                error_code = cls.__sspi_map.get(winerror, None)
            else:
                raise ValueError("base_error of type '%s' is not supported, must be a gssapi.exceptions.GSSError or WindowsError" % type(base_error).__name__)
        new_cls = cls.__registry.get(error_code or 0, cls)
        return super(_SpnegoErrorRegistry, new_cls).__call__(error_code, base_error, *args, **kwargs)