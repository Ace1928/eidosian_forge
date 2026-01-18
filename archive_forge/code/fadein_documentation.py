import numpy as np

    Makes the clip progressively appear from some color (black by default),
    over ``duration`` seconds at the beginning of the clip. Can be used for
    masks too, where the initial color must be a number between 0 and 1.
    For cross-fading (progressive appearance or disappearance of a clip
    over another clip, see ``composition.crossfade``
    