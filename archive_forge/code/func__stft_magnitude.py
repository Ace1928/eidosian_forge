import math
import torch
def _stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = _frame(signal, window_length, hop_length)
    window = torch.hann_window(window_length, periodic=True).to(signal.device)
    windowed_frames = frames * window
    return torch.abs(torch.fft.rfft(windowed_frames, int(fft_length)))