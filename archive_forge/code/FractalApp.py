import kivy
kivy.require('2.0.0')  # Specify your current Kivy version

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

# Ensure matplotlib does not use any Xwindows backend.
plt.switch_backend('agg')

# Fractal and system functions
def Y_total(t, G_params, k, I, lambda_val, n, epsilon, M_minus, threshold):
    a, b, c, d = G_params
    G = a*t - b*t**2 if t < c else c/(1 + d*np.exp(-t))
    G_inverse = -a*t + b*t**2 if t < c else -c/(1 + d*np.exp(-t))
    Delta_I = k * I
    return G * (1 + Delta_I) + lambda_val * n - epsilon if M_minus < threshold else G_inverse * (1 - Delta_I) - lambda_val * n - epsilon

def B(x, t, P):
    theta, f, g = P
    return np.sin(f * t) * np.cos(theta * x) * np.exp(-g * t)

def fractal_growth(x, t, system_params, fractal_params):
    G_params = (system_params['G_a'], system_params['G_b'], system_params['G_c'], system_params['G_d'])
    return Y_total(t, G_params, system_params['k'], system_params['I'], system_params['lambda'], system_params['n'], system_params['epsilon'], system_params['M_minus'], system_params['threshold']) * B(x, t, (fractal_params['theta'], fractal_params['f'], fractal_params['g']))

# Kivy Application Class
class FractalApp(App):
    def build(self):
        self.window = BoxLayout(orientation='vertical')
        
        # UI elements for parameters
        self.param_inputs = {}
        for param_name in ['G_a', 'G_b', 'G_c', 'G_d', 'k', 'I', 'lambda', 'n', 'epsilon', 'M_minus', 'threshold', 'theta', 'f', 'g']:
            box = BoxLayout(orientation='horizontal')
            label = Label(text=param_name)
            input_field = TextInput(text='', multiline=False)
            self.param_inputs[param_name] = input_field
            box.add_widget(label)
            box.add_widget(input_field)
            self.window.add_widget(box)

        # Update Button
        update_btn = Button(text="Update Fractal")
        update_btn.bind(on_press=self.update_fractal)
        self.window.add_widget(update_btn)

        # Image Widget for displaying fractal
        self.img = Image()
        self.window.add_widget(self.img)

        return self.window

    def update_fractal(self, instance):
        system_params = {}
        fractal_params = {}
        for key, value in self.param_inputs.items():
            try:
                if key in ['theta', 'f', 'g']:
                    fractal_params[key] = float(value.text)
                else:
                    system_params[key] = float(value.text)
            except ValueError:
                self.show_error(f"Invalid input for {key}")
                return

        x_range = np.linspace(-10, 10, 500)
        t_range = np.linspace(0, 10, 100)
        X, T = np.meshgrid(x_range, t_range)
        fractal_values = fractal_growth(X, T, system_params, fractal_params)

        self.plot_fractal_to_texture(fractal_values)

    def plot_fractal_to_texture(self, fractal_values):
        fig, ax = plt.subplots()
        cax = ax.imshow(fractal_values, extent=[-10, 10, 0, 10])
        fig.colorbar(cax)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        texture = Texture.create(size=(fig.canvas.get_width_height()), colorfmt='rgba')
        texture.blit_buffer(buf.getvalue(), colorfmt='rgba', bufferfmt='ubyte')
        buf.close()
        plt.close(fig)

        self.img.texture = texture

    def show_error(self, message):
        popup = Popup(title='Error', content=Label(text=message), size_hint=(None, None), size=(400, 400))
        popup.open()

# Entry point for the Kivy app
if __name__ == '__main__':
    FractalApp().run()
