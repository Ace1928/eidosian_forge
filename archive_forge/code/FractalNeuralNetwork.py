
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile

# Configure logging
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_file_path = "C:/Users/ace19/Documents/log_{}.txt".format(timestamp)
# Configure logging at the start of your script
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the logging for uncaught exceptions
sys.excepthook = log_exception

class SimulationParameters:
    def __init__(self):
        self.signal_processing_mode = 'complex'
        self.signal_amplification = 1.0
        self.pattern_detection_sensitivity = 1.0
        self.filter_params = {'filter_type': 'low_pass', 'cutoff_frequency': 100}
        self.wavelet_params = {'wavelet_type': 'db1', 'level': 2}
        self.pattern_params = {'max_score': 50}
        self.mapping_params = {'steepness': 1.0, 'skew_factor': 1.0}
        self.output_params = {'expansion_factor': 1.0}
        self.input_layer_neuron_count = 7  # Example value, set as per your network design
        self.neuron_signal_size = 6  # Example value, representing the size of the signal for each neuron

    def update_params(self, **kwargs):
        # Update parameters dynamically. Validate new values to ensure consistency and correctness.
        for key, value in kwargs.items():
            if key in self.__dict__:
                # Add validation for each parameter based on its expected type and range
                setattr(self, key, value)
            
class DynamicRingBuffer:
    def __init__(self, initial_size=1000):
        # Initialize a deque with a maximum length
        self.data = deque(maxlen=initial_size)
    def append(self, item):
        # Append an item to the deque; older items are automatically removed if max length is exceeded
        self.data.append(item)

    def get(self):
        # Return the contents of the deque as a list
        return list(self.data)
            
class SimplifiedNeuronWithDelay:
    @staticmethod
    def generate_random_param(mean, sigma):
        # Generate a random parameter clipped within a specific range (-128 to 127).
        # This function is used for initializing some neuron parameters.
        return np.clip(np.random.normal(mean, sigma), -128, 127)

    def __init__(self, threshold, over_threshold, base_signal_strength, refractory_period, spike_threshold, scaling_factor, neuron_state, damping_factor, num_sub_windows, amplification_factor, temporal_window_size, pattern_params, mapping_params, output_params, simulation_params):
        # Initialize neuron with various parameters. Input validation is performed to ensure data integrity.

        # Validate and assign neuron properties
        self.threshold = self._validate_param(threshold, 'Threshold', int, float, min_val=0, max_val=127)
        self.over_threshold = self._validate_param(over_threshold, 'Over Threshold', int, float, min_val=0, max_val=127)
        self.base_signal_strength = self._validate_param(base_signal_strength, 'Base Signal Strength', int, float, min_val=1, max_val=128)
        self.refractory_period = self._validate_param(refractory_period, 'Refractory Period', int, float, min_val=1, max_val=5)
        self.spike_threshold = self._validate_param(spike_threshold, 'Spike Threshold', int, float, min_val=1, max_val=127)
        self.scaling_factor = self._validate_param(scaling_factor, 'Scaling Factor', int, float)
        self.neuron_state = self._validate_choice(neuron_state, 'Neuron State', ['excitatory', 'inhibitory', 'neutral'])
        self.damping_factor = self._validate_param(damping_factor, 'Damping Factor', int, float, min_val=1)
        self.num_sub_windows = self._validate_param(num_sub_windows, 'Number of Sub-windows', int, min_val=1)
        self.amplification_factor = self._validate_param(amplification_factor, 'Amplification Factor', int, float, min_val=1)
        self.temporal_window_size = self._validate_param(temporal_window_size, 'Temporal Window Size', int, min_val=1)

        # Initialize deque for input history with a fixed size
        self.input_history = deque(maxlen=temporal_window_size)

        # Additional parameters related to neuron behavior
        self.last_spike_time = -1
        self.signal_processing_mode = simulation_params.signal_processing_mode
        self.pattern_params = pattern_params
        self.mapping_params = mapping_params
        self.output_params = output_params

        # Generate random parameters
        self.spike_magnitude = self.generate_random_param(30, 5)
        self.pattern_detection_threshold = self.generate_random_param(0.5, 0.1)
        self.mapping_steepness = self.generate_random_param(0.1, 0.02)
        self.mapping_skew_factor = self.generate_random_param(0.1, 0.02)

        # Validate the types of pattern_params, mapping_params, and output_params
        self.pattern_params = self._validate_param(pattern_params, 'Pattern Params', dict)
        self.mapping_params = self._validate_param(mapping_params, 'Mapping Params', dict)
        self.output_params = self._validate_param(output_params, 'Output Params', dict)

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        # Validate parameter with type checking and range validation if applicable
        if not isinstance(param, types):
            raise TypeError(f"{name} must be of type {types}")
        if min_val is not None and param < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and param > max_val:
            raise ValueError(f"{name} must not exceed {max_val}")
        return param

    def _validate_choice(self, choice, name, valid_choices):
        # Validate if the choice is among the valid options
        if choice not in valid_choices:
            raise ValueError(f"{name} must be one of {valid_choices}")
        return choice
       
    def advanced_signal_processing(self, signals, simulation_params):
        # Advanced signal processing logic based on simulation parameters.
        # This function should be implemented based on specific signal processing requirements.
        # Currently, it's a placeholder that simply doubles the signals.
        # Ensure 'signals' is a NumPy array and validate its dimensions and types.
        self._validate_signals(signals)
        # Here you can implement the actual advanced signal processing logic.
        return signals * 2

    def basic_signal_processing(self, signals, simulation_params):
        # Basic signal processing, such as a simple linear transformation.
        # Validate the signals before processing.
        self._validate_signals(signals)
        return signals * simulation_params.signal_amplification

    def spike_response(self):
        # Generate a response when the neuron spikes.
        # This could be a fixed value or computed based on neuron's state.
        return self.spike_magnitude

    def detect_pattern(self, input_history, processing_params):
        # Validate processing_params type.
        if not isinstance(processing_params, dict):
            raise TypeError("Processing parameters must be a dictionary")
        pattern_strength = self.calculate_pattern_strength(input_history, processing_params)
        return pattern_strength > self.pattern_detection_threshold * processing_params["pattern_detection_sensitivity"]

    def calculate_pattern_strength(self, input_history, processing_params):
        if len(input_history) == 0:
            return 0
        input_history_array = np.array(input_history)
        fft_result = np.fft.fft(input_history_array, axis=0)
        dominant_frequencies = np.abs(fft_result).mean(axis=0)
        entropy = -np.sum(dominant_frequencies * np.log(dominant_frequencies + 1e-9), axis=0)
        combined_score = np.sum(dominant_frequencies) + entropy
        normalized_score = np.clip(combined_score / processing_params["pattern_params"]["max_score"], 0, 1)

        # If normalized_score is an array, take the mean (or another aggregation method)
        # and then convert to an integer
        if isinstance(normalized_score, np.ndarray):
            normalized_score = int(np.mean(normalized_score))
        else:
            normalized_score = int(normalized_score)

        return normalized_score

    def state_mapping_function(self, input_signal_sum, mapping_params):
        # Default values for steepness and skew_factor
        default_steepness = 1.0
        default_skew_factor = 1.0

        steepness = self.mapping_steepness * mapping_params.get("steepness", default_steepness)
        skew_factor = self.mapping_skew_factor * mapping_params.get("skew_factor", default_skew_factor)

        sigmoid = 1 / (1 + np.exp(-steepness * (input_signal_sum - self.threshold)))
        skewed_value = np.log1p(abs(input_signal_sum)) ** skew_factor
        combined_value = sigmoid * skewed_value
        mapped_state = np.clip(int(combined_value * 6), -6, 6)

        return mapped_state

    def granular_output(self, state, output_params):
        # Generate a granular output based on the neuron's state.
        # This function scales the state based on an expansion factor.
        expanded_output = state * output_params["expansion_factor"]
        return expanded_output

    def process_input(self, input_signals, current_time, simulation_params):
        # Process input signals based on the current time and simulation parameters.
        # Extensive validation is performed to ensure data integrity.
        self._validate_process_input_params(input_signals, current_time, simulation_params)

        # Process signals based on the selected mode
        processed_signals = self._select_signal_processing_mode(input_signals, simulation_params)

        # Spike response logic
        if current_time - self.last_spike_time < self.refractory_period:
            return 0
        total_signal = np.sum(processed_signals) * simulation_params.signal_amplification
        if total_signal > self.spike_threshold:
            self.last_spike_time = current_time
            return self.spike_response()

        # Remaining processing logic
        scaled_signal = total_signal * self.scaling_factor
        damped_signal = scaled_signal // self.damping_factor
        self.input_history.append(input_signals)
        # Ensure correct processing parameters are passed to detect_pattern
        processing_params = {
            "pattern_detection_sensitivity": simulation_params.pattern_detection_sensitivity,
            "pattern_params": self.pattern_params
        }
        if self.detect_pattern(self.input_history, processing_params):
            damped_signal *= 1.2
        mapped_state = self.state_mapping_function(damped_signal, self.mapping_params)
        output = mapped_state * np.sign(total_signal) * self.amplification_factor
        return self.granular_output(output, self.output_params)

    # Helper methods for validation
    def _validate_signals(self, signals):
        # Validate that 'signals' is a NumPy array with appropriate dimensions.
        if not isinstance(signals, np.ndarray):
            raise TypeError("Signals must be a NumPy array")
        if signals.ndim != 2:
            raise ValueError("Signals array must be 2-dimensional")

    def _validate_process_input_params(self, input_signals, current_time, simulation_params):
        # Validate input parameters for the process_input method.
        self._validate_signals(input_signals)
        if not isinstance(current_time, (int, float)):
            raise TypeError("Current time must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

    def _select_signal_processing_mode(self, signals, simulation_params):
        # Select the appropriate signal processing mode based on simulation parameters.
        if simulation_params.signal_processing_mode in ['advanced', 'complex']:
            return self.advanced_signal_processing(signals, simulation_params)
        else:
            return self.basic_signal_processing(signals, simulation_params)
    
class ConnectionWithDelay:
    def __init__(self, strength, repeat_factor, invert_signal):
        # Initialize connection with delay parameters.
        # Validate each parameter for appropriate types and ranges.
        self.strength = self._validate_param(strength, 'Strength', float, int, min_val=0, max_val=127)
        self.repeat_factor = self._validate_param(repeat_factor, 'Repeat Factor', int, min_val=1, max_val=5)
        self.invert_signal = self._validate_param(invert_signal, 'Invert Signal', bool)

        # Create a deque to store delayed signals
        self.delayed_signals = deque([0] * self.repeat_factor, maxlen=self.repeat_factor)

    def transmit(self, signal, simulation_params):
        # Transmit a signal through the connection, applying delay and inversion if needed.
        # Validate signal and simulation_params.
        if not isinstance(signal, (int, float)):
            raise TypeError("Signal must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

        if self.invert_signal:
            signal = -signal
        repeated_signal = signal * self.strength
        self.delayed_signals.append(repeated_signal)
        return self.delayed_signals.popleft()

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        # Validate parameter with type checking and range validation if applicable.
        if not isinstance(param, types):
            raise TypeError(f"{name} must be of type {types}")
        if min_val is not None and param < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and param > max_val:
            raise ValueError(f"{name} must not exceed {max_val}")
        return param
    
class MiniNetworkWithDelays:
    def __init__(self, neurons, connections):
        # Initialize mini network with neurons and connections.
        # Validate neurons and connections.
        if not all(isinstance(neuron, SimplifiedNeuronWithDelay) for neuron in neurons):
            raise TypeError("All neurons must be instances of SimplifiedNeuronWithDelay")
        if not isinstance(connections, dict):
            raise TypeError("Connections must be a dictionary")

        self.neurons = neurons
        self.connections = connections

    def simulate(self, external_input, current_time, simulation_params):
        # Simulate network activity based on external input, current time, and simulation parameters.
        # Validate inputs.
        if not isinstance(external_input, np.ndarray):
            raise TypeError("External input must be a NumPy array")
        if not isinstance(current_time, (int, float)):
            raise TypeError("Current time must be a number")
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError("Simulation parameters must be an instance of SimulationParameters")

        neuron_outputs = [0] * len(self.neurons)

        # Process external input for the first six neurons
        for i in range(6):
            neuron_input = external_input[i] if i < len(external_input) else np.zeros((simulation_params.neuron_signal_size,))
            neuron_outputs[i] = self.neurons[i].process_input(neuron_input.reshape(1, -1), current_time, simulation_params)

        # Process internal connections and time window for all neurons
        for i in range(len(self.neurons)):
            aggregated_input = np.zeros((simulation_params.neuron_signal_size,))
            for j in range(len(self.neurons)):
                if (j, i) in self.connections:
                    connection = self.connections[(j, i)]
                    transmitted_signal = connection.transmit(neuron_outputs[j], simulation_params)
                    transmitted_signal = np.array([transmitted_signal])  # Ensure transmitted_signal is a NumPy array
                    aggregated_input += transmitted_signal.reshape(-1)

            # Combine external and internal inputs for the first six neurons
            if i < 6:
                neuron_output_array = np.array([neuron_outputs[i]])
                aggregated_input += neuron_output_array.reshape(-1)

            # Process input for each neuron
            neuron_outputs[i] = self.neurons[i].process_input(aggregated_input.reshape(1, -1), current_time, simulation_params)

        return neuron_outputs
    
def calculate_total_neurons(N, W):
    """
    Calculates the total number of neurons in a layered neural network.

    Parameters:
    N (int): Number of layers in the network.
    W (int): Width of each layer.

    Returns:
    int: Total number of neurons in the network.
    """
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer")
    if not isinstance(W, int) or W <= 0:
        raise ValueError("W must be a positive integer")

    neurons_base_layer = W * 7 * (6 ** (N - 1))
    neurons_above_base = W * sum(6 ** (N - i) for i in range(2, N + 1))
    total_neurons = neurons_base_layer + neurons_above_base
    return total_neurons

def random_neuron_params(simulation_params):
    # Generate random parameters for neuron based on simulation_params.
    # Validate that simulation_params is an instance of SimulationParameters.
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    # Retrieve 'max_score_range' from simulation_params, with a default value if not present.
    max_score_range = simulation_params.pattern_params.get('max_score_range', (10, 100))

    # Generate and return random parameters for a neuron.
    return {
        'threshold': np.clip(np.random.lognormal(mean=np.log(35), sigma=0.5), 0, 127),
        'over_threshold': np.clip(np.random.lognormal(mean=np.log(70), sigma=0.5), 0, 127),
        'base_signal_strength': np.clip(np.random.lognormal(mean=np.log(64), sigma=0.5), 1, 128),
        'refractory_period': np.clip(np.random.lognormal(mean=np.log(3), sigma=0.5), 1, 5),
        'spike_threshold': np.clip(np.random.lognormal(mean=np.log(50), sigma=0.5), 1, 127),
        'scaling_factor': np.random.uniform(0.1, 2.0),
        'neuron_state': random.choice(['excitatory', 'inhibitory', 'neutral']),
        'damping_factor': np.clip(np.random.lognormal(mean=np.log(2), sigma=0.3), 1, 5),
        'num_sub_windows': np.random.randint(1, 10),
        'amplification_factor': np.random.uniform(1, 5),
        'temporal_window_size': np.random.randint(5, 20),
        'pattern_params': {'max_score': np.random.uniform(*max_score_range)},
        'mapping_params': {
            'layer1_factor': np.random.uniform(0.1, 1.0),
            'layer2_exponent': np.random.uniform(1, 3),
            'steepness': np.random.uniform(0.1, 1.0)
        },
        'output_params': {'expansion_factor': np.random.uniform(1, 5)}
    }

def random_connection_params(simulation_params):
    # Generate random parameters for connections, influenced by simulation_params.
    # Validate that simulation_params is an instance of SimulationParameters.
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    # Generate and return random connection parameters, ensuring they are within valid ranges.
    return {
        'strength': np.clip(np.random.uniform(0.5, 1.0), 0, 127),  # Clipping to ensure strength is within the valid range.
        'repeat_factor': np.random.randint(1, 5),  # Repeat factor between 1 and 4.
        'invert_signal': random.choice([True, False])  # Randomly choose whether to invert the signal.
    }

def standard_input_transform(input_data, signal_size, num_neurons, simulation_params):
    # Transform standard input based on the type of input_data.
    # Validate input_data, signal_size, num_neurons, and simulation_params.
    if not isinstance(signal_size, int) or signal_size <= 0:
        raise ValueError("signal_size must be a positive integer")
    if not isinstance(num_neurons, int) or num_neurons <= 0:
        raise ValueError("num_neurons must be a positive integer")
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    # Logic for transforming array or text input.
    if isinstance(input_data, np.ndarray):
        return transform_array_input(input_data, signal_size, num_neurons)
    elif isinstance(input_data, str):
        return transform_text_input(input_data, signal_size, num_neurons)
    else:
        raise ValueError("Unsupported input type")

# Placeholder functions for transform_array_input and transform_text_input
def transform_array_input(array_input, signal_size, num_neurons):
    # Implement the logic to transform array input (e.g., flatten, resize, normalize)
    # This is a placeholder implementation.
    # Example: return np.reshape(array_input, (num_neurons, signal_size))
    return np.reshape(array_input, (num_neurons, signal_size))

def transform_text_input(text_input, signal_size, num_neurons):
    # Implement the logic to transform text input (e.g., encoding, padding)
    # This is a placeholder implementation.
    # Example: Convert text to numerical representation and resize.
    # return np.zeros((num_neurons, signal_size))  # Placeholder
    return np.zeros((num_neurons, signal_size))

def dynamic_input_generator(num_time_steps, simulation_params, external_input=None):
    # Generate dynamic input for a given number of time steps.
    # Validate num_time_steps and simulation_params.
    if not isinstance(num_time_steps, int) or num_time_steps <= 0:
        raise ValueError("num_time_steps must be a positive integer")
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")

    num_neurons = simulation_params.input_layer_neuron_count
    signal_size = simulation_params.neuron_signal_size

    # Generate transformed input based on whether external input is provided.
    if external_input is not None:
        transformed_input = standard_input_transform(external_input, signal_size, num_neurons, simulation_params)
    else:
        # Generate random input if no external input is provided.
        transformed_input = np.random.randint(-128, 128, size=(num_neurons, signal_size))

    # Replicate the transformed input for each time step.
    return np.array([transformed_input for _ in range(num_time_steps)])
    
def create_mini_network(simulation_params, connection_mode='specified'):
    # Validate simulation_params.
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError("simulation_params must be an instance of SimulationParameters")
    
    # Create neurons with simulation_params included.
    neuron_params = [random_neuron_params(simulation_params) for _ in range(7)]
    neurons = [SimplifiedNeuronWithDelay(**params, simulation_params=simulation_params) for params in neuron_params]

    # Define connections based on the connection mode.
    connections = {}
    if connection_mode == 'specified':
        specified_connections = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
            (1, 3), (3, 5), (5, 1), (2, 4), (4, 6), (6, 2),
            (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)
        ]
        for i, j in specified_connections:
            connections[(i, j)] = ConnectionWithDelay(**random_connection_params(simulation_params))
    elif connection_mode == 'other_mode':
        # Implement other connection modes.
        pass
    # ... additional connection modes can be added here

    return MiniNetworkWithDelays(neurons, connections)

def generate_new_data(existing_data, num_new_points, buffer_size=100):
    """
    Generates new random data for the neural network simulation.

    Parameters:
    existing_data (np.ndarray): Current data buffer of the simulation.
    num_new_points (int): Number of new data points to generate.
    buffer_size (int): Size of the data buffer to maintain.

    Returns:
    np.ndarray: Updated data buffer with new random data appended.
    """
    # Generate new random data of shape (num_new_points, 7)
    new_data = np.random.rand(num_new_points, 7)
    
    # Combine the existing data with the new data
    combined_data = np.vstack((existing_data, new_data))
    
    # If the combined data is larger than buffer size, trim it to keep the last 'buffer_size' rows
    if combined_data.shape[0] > buffer_size:
        return combined_data[-buffer_size:]
    else:
        return combined_data

def visualize_network_activity(initial_data):
    """
    Visualizes the neural network activity using Matplotlib.

    Parameters:
    initial_data (np.ndarray): The initial dataset for the simulation.
    """
    global ani
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Ensure initial_data is a NumPy array
    initial_data = np.array(initial_data) if not isinstance(initial_data, np.ndarray) else initial_data
    
    # Create initial plot lines
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    lines1 = [ax1.plot(initial_data[:, i], label=f'Neuron {i+1}', color=colors[i], linewidth=1)[0] for i in range(6)]
    lines2 = [ax2.plot(initial_data[:, 6], label='Node 7', color='k', linewidth=2)[0]]

    # Setup plot aesthetics
    ax1.set_title('Activity of Neurons 1-6')
    ax2.set_title('Activity of Node 7')
    for ax in [ax1, ax2]:
        ax.legend()
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activity')
        ax.grid(True)

    data_buffer = np.copy(initial_data)
    num_new_points = [1]  # Starting with 1 new point per frame

    dynamic_x_axis = [True]  # State variable for dynamic X-axis behavior
    x_axis_range = [20]  # Default range for the X-axis

    # UI elements need to be defined before the callbacks
     # Slider for controlling the simulation speed
    ax_slider_speed = plt.axes([0.25, 0.01, 0.50, 0.03], facecolor='lightgoldenrodyellow')
    speed_slider = Slider(ax_slider_speed, 'Speed', 1, 100, valinit=50)

    # Check button for toggling dynamic X-axis
    ax_check = plt.axes([0.8, 0.02, 0.15, 0.1], facecolor='lightgoldenrodyellow')
    dynamic_check = CheckButtons(ax_check, ['Dynamic X-Axis'], [True])

    # Reset button
    ax_button = plt.axes([0.05, 0.02, 0.1, 0.05])
    reset_button = Button(ax_button, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
    
    # Callback function for the X-axis range slider
    def update_x_range(val):
        nonlocal x_axis_range  # Use nonlocal to modify the outer variable
        x_axis_range[0] = int(val)
        # Update the plot directly if not animating
        if not dynamic_x_axis[0]:
            ax1.set_xlim(0, x_axis_range[0])
            ax2.set_xlim(0, x_axis_range[0])
            fig.canvas.draw_idle()


    # Callback function for the dynamic X-axis check button
    def toggle_dynamic(label):
        dynamic_x_axis[0] = not dynamic_x_axis[0]
        # Update the animation accordingly
        # ani.event_source.stop()
        # ani.event_source.start()

    # Callback function for the reset button
    def reset_to_default(event):
        dynamic_x_axis[0] = True
        x_axis_range[0] = 20
        dynamic_check.set_active(0)
        # Reset the animation view to default
        # ani.event_source.stop()
        # ani.event_source.start()
    # Connect the sliders and buttons to their callback functions
    dynamic_check.on_clicked(toggle_dynamic)
    reset_button.on_clicked(reset_to_default)
    
    def update_speed(val):
        """
        Update the animation speed based on the slider value.
        """
        ani.event_source.interval = 1000 / val  # Adjusting the interval of animation

    speed_slider.on_changed(update_speed)
    
    # Create the animation with FuncAnimation
    ani = FuncAnimation(fig, update_plot, fargs=(lines1 + lines2, data_buffer, num_new_points[0], ax1, ax2, 300, dynamic_x_axis[0], x_axis_range[0]), interval=100, blit=False)
    # Maximize the plot window
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')  # For TkAgg backend

    plt.show()
    # Ensure that the 'ani' variable is not collected by garbage collection
    # by making it global or attaching it to a higher scope object.
    ani = None
    
def update_plot(frame, lines, data_buffer, num_new_points, ax1, ax2, buffer_size=300, dynamic_x_axis=True, x_axis_range=20):
    """
    Update function for the animation. This function is called for each frame in the animation.

    Parameters:
    frame (int): Current frame number.
    lines (list): List of Line2D objects representing the data lines on the plot.
    data_buffer (np.ndarray): Buffer containing the data to be plotted.
    num_new_points (int): Number of new data points generated for each frame.
    ax1 (matplotlib.axes.Axes): Axes object for the first subplot.
    ax2 (matplotlib.axes.Axes): Axes object for the second subplot.
    buffer_size (int): Size of the data buffer.
    dynamic_x_axis (bool): Flag to determine if the x-axis should be dynamic.
    x_axis_range (int): Range of the x-axis.
    """
    try:
        # Generate new data and append it to the buffer
        new_data = generate_new_data(data_buffer[-min(len(data_buffer), buffer_size):], num_new_points)
        data_buffer = np.vstack((data_buffer, new_data))[-buffer_size:]

        # Update x_data for the plot based on the buffer size
        x_data = np.arange(start=max(0, len(data_buffer) - buffer_size), stop=len(data_buffer))

        # Update plot lines with new data
        for i, line in enumerate(lines):
            line.set_data(x_data, data_buffer[-buffer_size:, i])

        # Dynamically adjust the x-axis if enabled
        if dynamic_x_axis:
            x_min_current = max(0, len(data_buffer) - buffer_size)
            x_max_current = len(data_buffer)
        else:
            x_min_current = 0
            x_max_current = x_axis_range

        ax1.set_xlim(x_min_current, x_max_current)
        ax2.set_xlim(x_min_current, x_max_current)

        # Dynamic Y-Axis Adjustment for each Axes
        # For Neurons 1-6 (ax1)
        adjust_y_axis_limits(ax1, data_buffer[-buffer_size:, :6])

        # For Node 7 (ax2)
        adjust_y_axis_limits(ax2, data_buffer[-buffer_size:, 6])

        return lines

    except Exception as e:
        logging.exception("Exception occurred during plot update")
        raise

def adjust_y_axis_limits(ax, data, margin_factor=0.1):
    """
    Adjusts the y-axis limits dynamically based on the range of the data currently being visualized.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to adjust.
    data (np.ndarray): The data currently being visualized.
    margin_factor (float): A factor to determine the margin around the data range for better visibility.
    """
    y_min, y_max = np.min(data), np.max(data)
    y_margin = (y_max - y_min) * margin_factor
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    logging.debug(f"Y-axis for ax set to {y_min - y_margin}, {y_max + y_margin}")
    
def main(time_steps=100):
    """
    Main function to run the neural network simulation.

    Parameters:
    time_steps (int): Number of time steps for the simulation.
    """
    try:
        global ani  # Declare ani as global
        simulation_params = SimulationParameters()
        mini_network = create_mini_network(simulation_params)

        input_signals = dynamic_input_generator(time_steps, simulation_params)

        network_output = []
        for current_time in range(time_steps):
            output = mini_network.simulate(input_signals[current_time], current_time, simulation_params)
            network_output.append(output)

        ani = visualize_network_activity(network_output)  # Assign the result to ani once
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        print(f"An error occurred during the simulation: {e}")

if __name__ == "__main__":
    cProfile.run('main()')