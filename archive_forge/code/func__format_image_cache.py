import copy
import datetime
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _format_image_cache(cached_images):
    """Format image cache to make it more consistent with OSC operations."""
    image_list = []
    for item in cached_images:
        if item == 'cached_images':
            for image in cached_images[item]:
                image_obj = copy.deepcopy(image)
                image_obj['state'] = 'cached'
                image_obj['last_accessed'] = datetime.datetime.utcfromtimestamp(image['last_accessed']).isoformat()
                image_obj['last_modified'] = datetime.datetime.utcfromtimestamp(image['last_modified']).isoformat()
                image_list.append(image_obj)
        elif item == 'queued_images':
            for image in cached_images[item]:
                image = {'image_id': image}
                image.update({'state': 'queued', 'last_accessed': 'N/A', 'last_modified': 'N/A', 'size': 'N/A', 'hits': 'N/A'})
                image_list.append(image)
    return image_list