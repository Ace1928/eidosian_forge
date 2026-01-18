from collections import namedtuple
import re
import textwrap
import warnings
@classmethod
def _iterable_to_header_element(cls, iterable):
    """
        Convert iterable of tuples into header element ``str``.

        Each tuple is expected to be in one of two forms: (media_range, qvalue,
        extension_params_segment), or (media_range, qvalue).
        """
    try:
        media_range, qvalue, extension_params_segment = iterable
    except ValueError:
        media_range, qvalue = iterable
        extension_params_segment = ''
    if qvalue == 1.0:
        if extension_params_segment:
            element = '{};q=1{}'.format(media_range, extension_params_segment)
        else:
            element = media_range
    elif qvalue == 0.0:
        element = '{};q=0{}'.format(media_range, extension_params_segment)
    else:
        element = '{};q={}{}'.format(media_range, qvalue, extension_params_segment)
    return element