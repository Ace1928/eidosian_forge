import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import numpy as np
import matplotlib.pyplot as plt
import io
def Y_total(t, G_params, k, I, lambda_val, n, epsilon, M_minus, threshold):
    a, b, c, d = G_params
    G = a * t - b * t ** 2 if t < c else c / (1 + d * np.exp(-t))
    G_inverse = -a * t + b * t ** 2 if t < c else -c / (1 + d * np.exp(-t))
    Delta_I = k * I
    return G * (1 + Delta_I) + lambda_val * n - epsilon if M_minus < threshold else G_inverse * (1 - Delta_I) - lambda_val * n - epsilon