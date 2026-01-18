import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def _import_image_params(repo, tag, image=None, src=None, changes=None):
    params = {'repo': repo, 'tag': tag}
    if image:
        params['fromImage'] = image
    elif src and (not is_file(src)):
        params['fromSrc'] = src
    else:
        params['fromSrc'] = '-'
    if changes:
        params['changes'] = changes
    return params