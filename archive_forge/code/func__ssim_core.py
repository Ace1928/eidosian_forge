from ..utils import *
import numpy as np
import scipy.ndimage
def _ssim_core(referenceVideoFrame, distortedVideoFrame, K_1, K_2, bitdepth, scaleFix, avg_window):
    referenceVideoFrame = referenceVideoFrame.astype(np.float32)
    distortedVideoFrame = distortedVideoFrame.astype(np.float32)
    M, N = referenceVideoFrame.shape
    extend_mode = 'constant'
    if avg_window is None:
        avg_window = gen_gauss_window(5, 1.5)
    L = np.int(2 ** bitdepth - 1)
    C1 = (K_1 * L) ** 2
    C2 = (K_2 * L) ** 2
    factor = np.int(np.max((1, np.round(np.min((M, N)) / 256.0))))
    factor_lpf = np.ones((factor, factor), dtype=np.float32)
    factor_lpf /= np.sum(factor_lpf)
    if scaleFix:
        M = np.int(np.round(np.float(M) / factor + 1e-09))
        N = np.int(np.round(np.float(N) / factor + 1e-09))
    mu1 = np.zeros((M, N), dtype=np.float32)
    mu2 = np.zeros((M, N), dtype=np.float32)
    var1 = np.zeros((M, N), dtype=np.float32)
    var2 = np.zeros((M, N), dtype=np.float32)
    var12 = np.zeros((M, N), dtype=np.float32)
    if scaleFix and factor > 1:
        referenceVideoFrame = scipy.signal.correlate2d(referenceVideoFrame, factor_lpf, mode='same', boundary='symm')
        distortedVideoFrame = scipy.signal.correlate2d(distortedVideoFrame, factor_lpf, mode='same', boundary='symm')
        referenceVideoFrame = referenceVideoFrame[::factor, ::factor]
        distortedVideoFrame = distortedVideoFrame[::factor, ::factor]
    scipy.ndimage.correlate1d(referenceVideoFrame, avg_window, 0, mu1, mode=extend_mode)
    scipy.ndimage.correlate1d(mu1, avg_window, 1, mu1, mode=extend_mode)
    scipy.ndimage.correlate1d(distortedVideoFrame, avg_window, 0, mu2, mode=extend_mode)
    scipy.ndimage.correlate1d(mu2, avg_window, 1, mu2, mode=extend_mode)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    scipy.ndimage.correlate1d(referenceVideoFrame ** 2, avg_window, 0, var1, mode=extend_mode)
    scipy.ndimage.correlate1d(var1, avg_window, 1, var1, mode=extend_mode)
    scipy.ndimage.correlate1d(distortedVideoFrame ** 2, avg_window, 0, var2, mode=extend_mode)
    scipy.ndimage.correlate1d(var2, avg_window, 1, var2, mode=extend_mode)
    scipy.ndimage.correlate1d(referenceVideoFrame * distortedVideoFrame, avg_window, 0, var12, mode=extend_mode)
    scipy.ndimage.correlate1d(var12, avg_window, 1, var12, mode=extend_mode)
    sigma1_sq = var1 - mu1_sq
    sigma2_sq = var2 - mu2_sq
    sigma12 = var12 - mu1_mu2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_map[5:-5, 5:-5]
    cs_map = cs_map[5:-5, 5:-5]
    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)
    return (mssim, ssim_map, mcs, cs_map)