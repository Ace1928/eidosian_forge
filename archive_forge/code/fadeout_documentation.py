import numpy as np
from moviepy.decorators import requires_duration

    Makes the clip progressively fade to some color (black by default),
    over ``duration`` seconds at the end of the clip. Can be used for
    masks too, where the final color must be a number between 0 and 1.
    For cross-fading (progressive appearance or disappearance of a clip
    over another clip, see ``composition.crossfade``
    