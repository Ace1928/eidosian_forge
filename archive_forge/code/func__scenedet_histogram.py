import numpy as np
import scipy.ndimage
import scipy.spatial
from ..motion.gme import globalEdgeMotion
from ..utils import *
def _scenedet_histogram(videodata, parameter1, min_scene_len=2):
    detected_scenes = [0]
    numFrames, height, width, channels = videodata.shape
    for t in range(0, numFrames - 1):
        curr = rgb2gray(videodata[t])
        nxt = rgb2gray(videodata[t + 1])
        curr = curr[0, :, :, 0]
        nxt = nxt[0, :, :, 0]
        hist1, bins = np.histogram(curr, bins=256, range=(0, 255))
        hist2, bins = np.histogram(nxt, bins=256, range=(0, 255))
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        hist1 /= 256.0
        hist2 /= 256.0
        framediff = np.mean(np.abs(hist1 - hist2))
        if framediff > parameter1 and t - detected_scenes[len(detected_scenes) - 1] > min_scene_len:
            detected_scenes.append(t + 1)
    return np.array(detected_scenes)