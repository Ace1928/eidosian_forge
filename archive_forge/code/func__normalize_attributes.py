from ..exceptions import UndeclaredNamespace
@staticmethod
def _normalize_attributes(kv):
    k = kv[0].lower()
    v = k in ('rel', 'type') and kv[1].lower() or kv[1]
    return (k, v)