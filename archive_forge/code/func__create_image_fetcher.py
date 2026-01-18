import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def _create_image_fetcher():
    try:
        import pooch
        if not hasattr(pooch, '__version__'):
            retry = {}
        else:
            retry = {'retry_if_failed': 3}
    except ImportError:
        return (None, _LEGACY_DATA_DIR)
    if '+git' in __version__:
        skimage_version_for_pooch = __version__.replace('.dev0+git', '+git')
    else:
        skimage_version_for_pooch = __version__.replace('.dev', '+')
    if '+' in skimage_version_for_pooch:
        url = 'https://github.com/scikit-image/scikit-image/raw/{version}/skimage/'
    else:
        url = 'https://github.com/scikit-image/scikit-image/raw/v{version}/skimage/'
    image_fetcher = pooch.create(path=pooch.os_cache('scikit-image'), base_url=url, version=skimage_version_for_pooch, version_dev='main', env='SKIMAGE_DATADIR', registry=registry, urls=registry_urls, **retry)
    data_dir = osp.join(str(image_fetcher.abspath), 'data')
    return (image_fetcher, data_dir)