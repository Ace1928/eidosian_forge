import numpy as np
import scipy.ndimage
import scipy.spatial
from ..motion.gme import globalEdgeMotion
from ..utils import *
def _scenedet_edges(videodata, threshold, min_scene_len=2):
    detected_scenes = [0]
    r = 6
    luminancedata = rgb2gray(videodata)
    numFrames, height, width, channels = luminancedata.shape
    luminancedata = luminancedata[:, :, :, 0]
    for t in range(0, numFrames - 1):
        canny_in = canny(luminancedata[t])
        canny_out = canny(luminancedata[t + 1])
        disp = globalEdgeMotion(canny_in, canny_out)
        canny_out = np.roll(canny_out, disp[0], axis=0)
        canny_out = np.roll(canny_out, disp[1], axis=1)
        p_in = _percentage_distance(canny_in, canny_out, r)
        p_out = _percentage_distance(canny_out, canny_in, r)
        p = np.max((p_in, p_out))
        if p > threshold and t - detected_scenes[len(detected_scenes) - 1] > min_scene_len:
            detected_scenes.append(t + 1)
    return np.array(detected_scenes)