import collections
def _beta_metadatum(key, value):
    beta_key = key if isinstance(key, (bytes,)) else key.encode('ascii')
    beta_value = value if isinstance(value, (bytes,)) else value.encode('ascii')
    return _Metadatum(beta_key, beta_value)