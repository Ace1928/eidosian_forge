import logging
import six
def _parse_parameter_string(string, uri):
    for param_string in string.split('&'):
        try:
            yield _Parameter.from_string(param_string)
        except _InvalidParameter:
            logger.error("Invalid parameter '{param}' in capability URI '{uri}'".format(param=param_string, uri=uri))