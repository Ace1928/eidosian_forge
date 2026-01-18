import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def _scan_plugins():
    """Scan the plugins directory for .ini files and parse them
    to gather plugin meta-data.
    """
    pd = os.path.dirname(__file__)
    config_files = glob(os.path.join(pd, '_plugins', '*.ini'))
    for filename in config_files:
        name, meta_data = _parse_config_file(filename)
        if 'provides' not in meta_data:
            warnings.warn(f'file {filename} not recognized as a scikit-image io plugin, skipping.')
            continue
        plugin_meta_data[name] = meta_data
        provides = [s.strip() for s in meta_data['provides'].split(',')]
        valid_provides = [p for p in provides if p in plugin_store]
        for p in provides:
            if p not in plugin_store:
                print(f'Plugin `{name}` wants to provide non-existent `{p}`. Ignoring.')
        need_to_add_collection = 'imread_collection' not in valid_provides and 'imread' in valid_provides
        if need_to_add_collection:
            valid_provides.append('imread_collection')
        plugin_provides[name] = valid_provides
        plugin_module_name[name] = os.path.basename(filename)[:-4]