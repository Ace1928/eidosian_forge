import collections
def _metadatum(beta_key, beta_value):
    key = beta_key if isinstance(beta_key, (str,)) else beta_key.decode('utf8')
    if isinstance(beta_value, (str,)) or key[-4:] == '-bin':
        value = beta_value
    else:
        value = beta_value.decode('utf8')
    return _Metadatum(key, value)