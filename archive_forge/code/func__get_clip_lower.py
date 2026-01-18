import numpy as np
from scipy import stats
def _get_clip_lower(self, kwds):
    """helper method to get clip_lower from kwds or attribute

        """
    if 'clip_lower' not in kwds:
        clip_lower = self.clip_lower
    else:
        clip_lower = kwds.pop('clip_lower')
    return (clip_lower, kwds)