import os
import numpy as np
import proglog
from tqdm import tqdm
from moviepy.audio.io.ffmpeg_audiowriter import ffmpeg_audiowrite
from moviepy.Clip import Clip
from moviepy.decorators import requires_duration
from moviepy.tools import deprecated_version_of, extensions_dict
class AudioArrayClip(AudioClip):
    """
    
    An audio clip made from a sound array.
    
    Parameters
    -----------
    
    array
      A Numpy array representing the sound, of size Nx1 for mono,
      Nx2 for stereo.
       
    fps
      Frames per second : speed at which the sound is supposed to be
      played.
    
    """

    def __init__(self, array, fps):
        Clip.__init__(self)
        self.array = array
        self.fps = fps
        self.duration = 1.0 * len(array) / fps

        def make_frame(t):
            """ complicated, but must be able to handle the case where t
            is a list of the form sin(t) """
            if isinstance(t, np.ndarray):
                array_inds = (self.fps * t).astype(int)
                in_array = (array_inds > 0) & (array_inds < len(self.array))
                result = np.zeros((len(t), 2))
                result[in_array] = self.array[array_inds[in_array]]
                return result
            else:
                i = int(self.fps * t)
                if i < 0 or i >= len(self.array):
                    return 0 * self.array[0]
                else:
                    return self.array[i]
        self.make_frame = make_frame
        self.nchannels = len(list(self.get_frame(0)))