from typing import Optional
def _set_config_path(path: Optional[str]=None):
    """
    The function is used by the library to provide the local
    path of the code path to the users so it can be referenced
    while loading the chain back.
    """
    globals()['__databricks_rag_config_path__'] = path