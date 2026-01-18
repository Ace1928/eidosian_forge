"""
To design a highly sophisticated Digital Audio Workstation (DAW) that seamlessly integrates all 20 sound synthesis modules, with provisions for dynamic loading and robust error handling, we will create an advanced, modular, and extensible system using Python. This system will leverage the PyQt framework for the GUI, enabling real-time user interaction, and PyAudio for handling audio streams efficiently. Each module will be designed to operate independently, ensuring that the system remains functional even if some modules fail to load.

### 1. Sound Module Base Class
This base class will define a standard interface for all sound processing modules, ensuring uniformity and facilitating easier maintenance and enhancements.
"""

import numpy as np
from scipy.signal import resample, fftconvolve
from typing import Union

SAMPLE_RATE = 44100  # Default sample rate for audio processing


class SoundModule:
    """
    Abstract base class for all sound modules in the DAW.
    This class defines the interface and common functionality across all sound modules.
    """

    def __init__(self):
        pass

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Process the sound data. Must be implemented by each module to modify the audio signal.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data.
        """
        raise NotImplementedError(
            "Each module must implement the process_sound method."
        )

    def set_parameter(self, parameter: str, value: float):
        """
        Set parameters for the sound module. Should be implemented by modules that have parameters.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.
        """
        raise NotImplementedError(
            "This method should be overridden by modules that have parameters."
        )


"""
### 2. Specific Sound Module Implementations
Each sound synthesis module will inherit from `SoundModule` and implement its specific functionality, such as Amplitude Control and Envelope Generator. For simplicity, we illustrate two modules:
"""


class AmplitudeControl(SoundModule):
    """
    Controls the amplitude of the sound. This class inherits from the SoundModule base class and
    provides specific functionality to adjust the volume of the sound dynamically.

    Attributes:
        volume (float): The current volume level of the sound. This value is a floating-point number
                        where 1.0 represents the original amplitude, less than 1.0 represents a decrease
                        in amplitude, and greater than 1.0 represents an increase in amplitude.

    Parameters:
        initial_volume (float): The initial volume of the sound. Default is 1.0, which means no change
                                to the input sound's amplitude.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Dynamically applies amplitude control to the input sound based on the current volume setting.
        set_parameter(parameter: str, value: float) -> None:
            Allows dynamic adjustment of the module's parameters. Currently supports the 'volume' parameter.
    """

    def __init__(self, initial_volume: float = 1.0) -> None:
        """
        Initializes the AmplitudeControl module with the specified initial volume.

        Args:
            initial_volume (float): The initial volume level for sound processing. Defaults to 1.0.
        """
        super().__init__()
        self.volume: float = initial_volume  # Set the initial volume of the module.

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Processes the input sound data by applying amplitude control based on the current volume setting.

        Args:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after applying amplitude control. The amplitude of the sound is
                        adjusted by multiplying the sound data by the current volume level.
        """
        # Ensure the input sound is a NumPy array for proper processing.
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Apply amplitude control by scaling the sound array with the current volume.
        processed_sound: np.ndarray = sound * self.volume
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported. Currently, this method
        supports adjusting the 'volume' parameter.

        Args:
            parameter (str): The name of the parameter to set. Supported parameter: 'volume'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is out of the expected range.
        """
        if parameter == "volume":
            if not (0.0 <= value <= 2.0):
                raise ValueError("Volume must be between 0.0 and 2.0.")
            self.volume = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class EnvelopeGenerator(SoundModule):
    """
    Generates an ADSR (Attack, Decay, Sustain, Release) envelope for sound shaping. This class inherits from the SoundModule base class and provides specific functionality to dynamically shape the amplitude of a sound signal over time according to the ADSR envelope parameters.

    Attributes:
        attack (float): The attack time of the envelope in seconds, defining how quickly the sound reaches its peak amplitude.
        decay (float): The decay time of the envelope in seconds, defining how quickly the sound reduces to the sustain level after the initial peak.
        sustain (float): The sustain level of the envelope, representing the amplitude level during the main sequence of the sound's duration, before the release starts.
        release (float): The release time of the envelope in seconds, defining how quickly the sound fades out after the sustain phase.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the ADSR envelope to the input sound based on the current envelope parameters.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value if it is supported.
            Supported parameters: 'attack', 'decay', 'sustain', 'release'.
    """

    def __init__(
        self, attack: float, decay: float, sustain: float, release: float
    ) -> None:
        """
        Initializes the EnvelopeGenerator module with the specified ADSR parameters.

        Args:
            attack (float): The attack time in seconds.
            decay (float): The decay time in seconds.
            sustain (float): The sustain level (0.0 to 1.0).
            release (float): The release time in seconds.
        """
        super().__init__()
        self.attack: float = attack
        self.decay: float = decay
        self.sustain: float = sustain
        self.release: float = release

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the ADSR envelope to the input sound. This method modifies the amplitude of the sound data over time according to the ADSR parameters.

        Args:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after applying the ADSR envelope.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Create an envelope curve based on the ADSR parameters
        num_samples = len(sound)
        attack_samples = int(self.attack * SAMPLE_RATE)
        decay_samples = int(self.decay * SAMPLE_RATE)
        release_samples = int(self.release * SAMPLE_RATE)
        sustain_samples = num_samples - (
            attack_samples + decay_samples + release_samples
        )

        # Ensure that the sustain phase has at least one sample
        if sustain_samples < 1:
            raise ValueError(
                "ADSR envelope parameters do not allow for a valid sustain phase with the given sound length."
            )

        # Generate the envelope
        envelope = np.concatenate(
            [
                np.linspace(0, 1, attack_samples),  # Attack phase
                np.linspace(1, self.sustain, decay_samples),  # Decay phase
                np.full(sustain_samples, self.sustain),  # Sustain phase
                np.linspace(self.sustain, 0, release_samples),  # Release phase
            ]
        )

        # Apply the envelope to the sound
        processed_sound = sound * envelope
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Args:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is out of the expected range.
        """
        if parameter in ["attack", "decay", "sustain", "release"]:
            if not (
                0.0 <= value <= 10.0
            ):  # Assuming reasonable limits for ADSR parameters
                raise ValueError(
                    f"{parameter.capitalize()} must be between 0.0 and 10.0."
                )
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class PitchControl(SoundModule):
    """
    Manages the pitch alterations of a sound by adjusting its frequency.

    Attributes:
        base_frequency (float): The base frequency of the sound in Hz, which serves as the reference point for pitch adjustments.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies pitch control to the input sound by modifying its frequency.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value if it is supported.
            Supported parameters: 'base_frequency'.
    """

    def __init__(self, base_frequency: float) -> None:
        """
        Initializes the PitchControl module with a specified base frequency.

        Args:
            base_frequency (float): The base frequency in Hz for the pitch control.
        """
        super().__init__()
        self.base_frequency: float = base_frequency

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies pitch control to the input sound by adjusting its frequency relative to the base frequency.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with pitch control applied, adjusting the pitch based on the base frequency.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Calculate the factor by which to change the sample rate to achieve the desired pitch
        current_frequency = np.fft.rfftfreq(sound.size, d=1 / SAMPLE_RATE)
        if current_frequency.size == 0:
            raise ValueError("Input sound is too short to determine its frequency.")

        # Determine the dominant frequency in the sound
        dominant_frequency = current_frequency[np.argmax(np.abs(np.fft.rfft(sound)))]
        pitch_shift_factor = self.base_frequency / dominant_frequency

        # Resample the sound to achieve the pitch shift
        new_length = int(sound.size / pitch_shift_factor)
        processed_sound = resample(sound, new_length)

        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'base_frequency'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is not a valid float.
        """
        if parameter == "base_frequency":
            if not isinstance(value, float):
                raise ValueError("Base frequency must be a float.")
            self.base_frequency = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class TimbreAdjustment(SoundModule):
    """
    Adjusts the timbre or tone color of a sound by manipulating its harmonic content. This class inherits from the SoundModule base class and provides specific functionality to modify the harmonic content of the sound based on the specified harmonics.

    Attributes:
        harmonics (dict): A dictionary representing the harmonics and their amplitudes, where keys are harmonic frequencies (in Hz) and values are amplitude multipliers.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies timbre adjustment to the input sound by modifying its harmonic content based on the specified harmonics.
        set_parameter(parameter: str, value: dict) -> None:
            Sets the specified parameter to the given value if it is supported.
            Supported parameters: 'harmonics'.
    """

    def __init__(self, harmonics: dict) -> None:
        """
        Initializes the TimbreAdjustment module with the specified harmonics.

        Args:
            harmonics (dict): A dictionary of harmonic frequencies and their corresponding amplitude multipliers.
        """
        super().__init__()
        if not isinstance(harmonics, dict):
            raise TypeError("Harmonics must be provided as a dictionary.")
        self.harmonics: dict = harmonics

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies timbre adjustment to the input sound by modifying its harmonic content.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with timbre adjustment applied.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Perform a Fourier transform on the input sound
        frequency_spectrum = np.fft.rfft(sound)
        frequencies = np.fft.rfftfreq(len(sound), d=1 / SAMPLE_RATE)

        # Adjust the amplitude of each harmonic in the frequency spectrum
        adjusted_spectrum = np.array(
            [
                amplitude * self.harmonics.get(freq, 1)
                for freq, amplitude in zip(frequencies, frequency_spectrum)
            ]
        )

        # Perform an inverse Fourier transform to convert back to time domain
        processed_sound = np.fft.irfft(adjusted_spectrum, n=len(sound))
        return processed_sound

    def set_parameter(self, parameter: str, value: dict) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'harmonics'.
            value (dict): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is not a dictionary.
        """
        if parameter == "harmonics":
            if not isinstance(value, dict):
                raise ValueError("Harmonics must be provided as a dictionary.")
            self.harmonics = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class HarmonicGenerator(SoundModule):
    """
    Generates and manipulates overtones above the fundamental frequency.

    This class inherits from the SoundModule base class and provides specific functionality
    to generate harmonics based on a fundamental frequency. It allows for dynamic manipulation
    of the harmonic content of a sound signal.

    Attributes:
        fundamental_frequency (float): The fundamental frequency of the sound in Hz.
        overtones (dict): A dictionary where keys are harmonic indices (integers) and values
                          are amplitude multipliers (floats).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies harmonic generation to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'fundamental_frequency'.
    """

    def __init__(self, fundamental_frequency: float) -> None:
        """
        Initializes the HarmonicGenerator module with the specified fundamental frequency.

        Args:
            fundamental_frequency (float): The fundamental frequency in Hz.
        """
        super().__init__()
        self.fundamental_frequency: float = fundamental_frequency
        self.overtones: dict = {
            1: 1.0
        }  # Initialize with the fundamental frequency only.

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies harmonic generation to the input sound by adding overtones based on the fundamental frequency.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with harmonic generation applied.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Perform a Fourier transform on the input sound
        frequency_spectrum = np.fft.rfft(sound)
        frequencies = np.fft.rfftfreq(len(sound), d=1 / SAMPLE_RATE)

        # Generate harmonics based on the fundamental frequency
        harmonic_spectrum = np.zeros_like(frequency_spectrum)
        for harmonic_index, amplitude_multiplier in self.overtones.items():
            harmonic_freq = harmonic_index * self.fundamental_frequency
            closest_freq_index = np.argmin(np.abs(frequencies - harmonic_freq))
            harmonic_spectrum[closest_freq_index] += (
                amplitude_multiplier * frequency_spectrum[closest_freq_index]
            )

        # Perform an inverse Fourier transform to convert back to time domain
        processed_sound = np.fft.irfft(harmonic_spectrum, n=len(sound))
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'fundamental_frequency'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is not a float.
        """
        if parameter == "fundamental_frequency":
            if not isinstance(value, float):
                raise ValueError("Fundamental frequency must be a float.")
            self.fundamental_frequency = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class ModulationTechniques(SoundModule):
    """
    Implements modulation techniques such as Amplitude Modulation (AM), Frequency Modulation (FM),
    and Phase Modulation (PM) to a sound. This class inherits from the SoundModule base class and
    provides specific functionality to modulate the sound based on the specified parameters.

    Attributes:
        modulation_type (str): The type of modulation to apply ('AM', 'FM', or 'PM').
        modulation_frequency (float): The frequency of the modulation signal in Hz.
        modulation_depth (float): The depth of the modulation effect (0.0 to 1.0).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the specified modulation technique to the input sound.
        set_parameter(parameter: str, value: Union[str, float]) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'modulation_type', 'modulation_frequency', 'modulation_depth'.
    """

    def __init__(
        self, modulation_type: str, modulation_frequency: float, modulation_depth: float
    ) -> None:
        """
        Initializes the ModulationTechniques module with the specified modulation type, frequency, and depth.

        Args:
            modulation_type (str): The type of modulation to apply ('AM', 'FM', or 'PM').
            modulation_frequency (float): The frequency of the modulation signal in Hz.
            modulation_depth (float): The depth of the modulation effect (0.0 to 1.0).
        """
        super().__init__()
        self.modulation_type: str = modulation_type
        self.modulation_frequency: float = modulation_frequency
        self.modulation_depth: float = modulation_depth

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the specified modulation technique to the input sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the modulation technique applied.

        Raises:
            ValueError: If the modulation type is unsupported.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Ensure the sound is one-dimensional for processing
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Create a time array
        t = np.arange(len(sound)) / SAMPLE_RATE

        # Modulation based on the type
        if self.modulation_type == "AM":
            # Amplitude Modulation
            carrier = np.sin(2 * np.pi * self.modulation_frequency * t)
            processed_sound = (1 + self.modulation_depth * carrier) * sound
        elif self.modulation_type == "FM":
            # Frequency Modulation
            carrier = np.sin(
                2 * np.pi * self.modulation_frequency * t
                + self.modulation_depth
                * np.sin(2 * np.pi * self.modulation_frequency * t)
            )
            processed_sound = sound * carrier
        elif self.modulation_type == "PM":
            # Phase Modulation
            carrier = np.sin(
                2 * np.pi * self.modulation_frequency * t
                + self.modulation_depth * np.sin(2 * np.pi * t)
            )
            processed_sound = sound * carrier
        else:
            raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

        return processed_sound

    def set_parameter(self, parameter: str, value: Union[str, float]) -> None:
        """
        Sets the specified parameter to the given value.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'modulation_type', 'modulation_frequency', 'modulation_depth'.
            value (Union[str, float]): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value type is incorrect.
        """
        if parameter == "modulation_type" and isinstance(value, str):
            self.modulation_type = value
        elif parameter == "modulation_frequency" and isinstance(value, float):
            self.modulation_frequency = value
        elif parameter == "modulation_depth" and isinstance(value, float):
            self.modulation_depth = value
        else:
            raise ValueError(f"Unsupported parameter or incorrect type: {parameter}")


class ReverbEffect(SoundModule):
    """
    Simulates reverberation effects mimicking sound reflections in various environments.

    Parameters:
        decay (float): The decay time of the reverb effect in seconds.
        pre_delay (float): The pre-delay time of the reverb effect in seconds.
        mix (float): The mix ratio between the original and reverberated sound (0.0 to 1.0).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the reverb effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'decay', 'pre_delay', 'mix'.
    """

    def __init__(self, decay: float, pre_delay: float, mix: float) -> None:
        """
        Initializes the ReverbEffect module with specified decay, pre-delay, and mix parameters.

        Args:
            decay (float): The decay time in seconds.
            pre_delay (float): The pre-delay time in seconds.
            mix (float): The mix ratio between the original and reverberated sound.
        """
        super().__init__()
        self.decay: float = decay
        self.pre_delay: float = pre_delay
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the reverb effect to the input sound using convolution with an exponentially decaying noise signal.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the reverb effect applied.
        """
        # Generate an impulse response for the reverb effect
        impulse_response = self._generate_impulse_response(len(sound))

        # Apply convolution to simulate the reverb effect
        reverberated_sound = fftconvolve(sound, impulse_response, mode="full")[
            : len(sound)
        ]

        # Mix the original sound with the reverberated sound
        processed_sound = (1 - self.mix) * sound + self.mix * reverberated_sound
        return processed_sound

    def _generate_impulse_response(self, length: int) -> np.ndarray:
        """
        Generates an impulse response using an exponentially decaying noise signal.

        Parameters:
            length (int): The length of the impulse response.

        Returns:
            np.ndarray: The generated impulse response.
        """
        t = np.linspace(0, self.decay, length)
        impulse_response = np.exp(-t / self.decay) * np.random.randn(length)
        return impulse_response

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'decay', 'pre_delay', 'mix'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is out of the expected range.
        """
        if parameter in ["decay", "pre_delay", "mix"]:
            if not (0.0 <= value <= 10.0):
                raise ValueError(
                    f"{parameter.capitalize()} must be between 0.0 and 10.0."
                )
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class EchoEffect(SoundModule):
    """
    Generates echo effects by delaying and replaying the sound.

    Parameters:
        delay_time (float): The delay time between echoes in seconds.
        feedback (float): The feedback amount controlling the echo strength (0.0 to 1.0).
        mix (float): The mix ratio between the original and echoed sound (0.0 to 1.0).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the echo effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'delay_time', 'feedback', 'mix'.
    """

    def __init__(self, delay_time: float, feedback: float, mix: float) -> None:
        """
        Initializes the EchoEffect module with specified delay time, feedback, and mix parameters.

        Args:
            delay_time (float): The delay time between echoes in seconds.
            feedback (float): The feedback amount controlling the echo strength.
            mix (float): The mix ratio between the original and echoed sound.
        """
        super().__init__()
        self.delay_time: float = delay_time
        self.feedback: float = feedback
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the echo effect to the input sound by creating a delayed version of the sound and mixing it back with the original sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the echo effect applied.
        """
        # Calculate the number of samples to delay
        delay_samples = int(self.delay_time * SAMPLE_RATE)

        # Create an empty array to store the echoed sound
        echoed_sound = np.zeros_like(sound)

        # Apply the echo effect
        for i in range(delay_samples, len(sound)):
            echoed_sound[i] = sound[i] + self.feedback * echoed_sound[i - delay_samples]

        # Mix the original sound with the echoed sound
        processed_sound = (1 - self.mix) * sound + self.mix * echoed_sound
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is out of the expected range.
        """
        if parameter in ["delay_time", "feedback", "mix"]:
            if parameter == "delay_time" and not (0.0 <= value <= 10.0):
                raise ValueError("Delay time must be between 0.0 and 10.0 seconds.")
            if parameter == "feedback" and not (0.0 <= value <= 1.0):
                raise ValueError("Feedback must be between 0.0 and 1.0.")
            if parameter == "mix" and not (0.0 <= value <= 1.0):
                raise ValueError("Mix must be between 0.0 and 1.0.")
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class ChorusEffect(SoundModule):
    """
    Implements a chorus effect to create a richer, thicker sound by duplicating the input sound and modulating the delay time of the copies.

    Attributes:
        rate (float): The modulation rate of the chorus effect in Hz.
        depth (float): The depth of the chorus effect, representing the maximum delay variation (0.0 to 1.0).
        mix (float): The mix ratio between the original and chorused sound (0.0 to 1.0).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the chorus effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value if it is supported.
            Supported parameters: 'rate', 'depth', 'mix'.
    """

    def __init__(self, rate: float, depth: float, mix: float) -> None:
        """
        Initializes the ChorusEffect module with specified modulation rate, depth, and mix ratio.

        Parameters:
            rate (float): The modulation rate of the chorus effect in Hz.
            depth (float): The depth of the chorus effect, representing the maximum delay variation.
            mix (float): The mix ratio between the original and chorused sound.
        """
        super().__init__()
        self.rate: float = rate
        self.depth: float = depth
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the chorus effect to the input sound by modulating the delay time of the sound copies.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the chorus effect applied.
        """
        num_samples = len(sound)
        max_delay_samples = int(self.depth * SAMPLE_RATE)
        delay_buffer = np.zeros((max_delay_samples,))
        output = np.zeros_like(sound)

        # Modulate delay time using a sine wave
        modulator = np.sin(2 * np.pi * np.arange(num_samples) * self.rate / SAMPLE_RATE)

        for i in range(num_samples):
            modulated_delay = int((modulator[i] * 0.5 + 0.5) * max_delay_samples)
            delay_index = (i - modulated_delay + max_delay_samples) % max_delay_samples

            # Store current sample in delay buffer
            delay_buffer[delay_index] = sound[i]

            # Mix original and delayed sound
            output[i] = (1 - self.mix) * sound[i] + self.mix * delay_buffer[delay_index]

        return output

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'rate', 'depth', 'mix'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or the value is out of the expected range.
        """
        if parameter in ["rate", "depth", "mix"]:
            if parameter == "rate" and not (0.01 <= value <= 10.0):
                raise ValueError("Rate must be between 0.01 Hz and 10.0 Hz.")
            if parameter == "depth" and not (0.0 <= value <= 1.0):
                raise ValueError("Depth must be between 0.0 and 1.0.")
            if parameter == "mix" and not (0.0 <= value <= 1.0):
                raise ValueError("Mix must be between 0.0 and 1.0.")
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class FlangerEffect(SoundModule):
    """
    Creates a flanging effect by mixing the sound with a delayed version of itself, modulated over time.

    Attributes:
        delay (float): The base delay time in seconds.
        depth (float): The depth of the flanging effect (0.0 to 1.0).
        rate (float): The modulation rate of the flanging effect in Hz.
        feedback (float): The feedback amount controlling the flanging strength (0.0 to 1.0).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the flanger effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'delay', 'depth', 'rate', 'feedback'.
    """

    def __init__(
        self, delay: float, depth: float, rate: float, feedback: float
    ) -> None:
        """
        Initializes the FlangerEffect module with the specified parameters.

        Args:
            delay (float): The base delay time in seconds.
            depth (float): The depth of the flanging effect.
            rate (float): The modulation rate of the flanging effect in Hz.
            feedback (float): The feedback amount controlling the flanging strength.
        """
        super().__init__()
        self.delay: float = delay
        self.depth: float = depth
        self.rate: float = rate
        self.feedback: float = feedback

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the flanger effect to the input sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the flanger effect applied.
        """
        num_samples = len(sound)
        delay_samples = int(self.delay * SAMPLE_RATE)
        max_delay_samples = int((self.delay + self.depth) * SAMPLE_RATE)
        delay_buffer = np.zeros((max_delay_samples,))
        output = np.zeros_like(sound)

        # Modulate delay time using a sine wave
        modulator = np.sin(2 * np.pi * np.arange(num_samples) * self.rate / SAMPLE_RATE)

        for i in range(num_samples):
            modulated_delay = int(
                delay_samples + self.depth * delay_samples * modulator[i]
            )
            delay_index = (i - modulated_delay + max_delay_samples) % max_delay_samples

            # Store current sample in delay buffer with feedback
            delay_buffer[delay_index] = (
                sound[i] + self.feedback * delay_buffer[delay_index]
            )

            # Mix original and delayed sound
            output[i] = (1 - self.mix) * sound[i] + self.mix * delay_buffer[delay_index]

        return output

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'delay', 'depth', 'rate', 'feedback'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or the value is out of the expected range.
        """
        if parameter in ["delay", "depth", "rate", "feedback"]:
            if parameter == "delay" and not (0.001 <= value <= 0.1):
                raise ValueError("Delay must be between 0.001 and 0.1 seconds.")
            if parameter == "depth" and not (0.0 <= value <= 1.0):
                raise ValueError("Depth must be between 0.0 and 1.0.")
            if parameter == "rate" and not (0.1 <= value <= 5.0):
                raise ValueError("Rate must be between 0.1 Hz and 5.0 Hz.")
            if parameter == "feedback" and not (0.0 <= value <= 1.0):
                raise ValueError("Feedback must be between 0.0 and 1.0.")
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class PhaserEffect(SoundModule):
    """
    Creates a phaser effect by filtering the sound to create peaks and troughs through phase modulation.

    Parameters:
        rate (float): The modulation rate of the phaser effect in Hz.
        depth (float): The depth of the phaser effect, ranging from 0.0 (no effect) to 1.0 (full effect).
        feedback (float): The feedback amount controlling the intensity of the phaser effect, ranging from 0.0 (no feedback) to 1.0 (maximum feedback).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the phaser effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'rate', 'depth', 'feedback'.
    """

    def __init__(self, rate: float, depth: float, feedback: float) -> None:
        """
        Initializes the PhaserEffect module with the specified modulation rate, depth, and feedback.

        Args:
            rate (float): The modulation rate in Hz.
            depth (float): The modulation depth.
            feedback (float): The feedback amount.
        """
        super().__init__()
        self.rate: float = rate
        self.depth: float = depth
        self.feedback: float = feedback

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the phaser effect to the input sound using a series of all-pass filters.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after the phaser effect has been applied.
        """
        num_samples = len(sound)
        # Create an LFO (Low Frequency Oscillator) to modulate the phase delay
        lfo = np.sin(2 * np.pi * np.arange(num_samples) * self.rate / SAMPLE_RATE)
        # Initialize the output array
        output = np.zeros_like(sound)
        # Initialize the all-pass filter delay buffer
        ap_delay_buffer = np.zeros(int(SAMPLE_RATE / self.rate))

        for i in range(num_samples):
            # Calculate the modulated delay
            modulated_delay = int(
                self.depth * SAMPLE_RATE / self.rate * (1 + lfo[i]) / 2
            )
            delay_index = i - modulated_delay

            # Wrap the delay index around the buffer size
            if delay_index < 0:
                delay_index += len(ap_delay_buffer)

            # Apply the all-pass filter
            ap_output = (
                -self.feedback * ap_delay_buffer[delay_index]
                + sound[i]
                + self.feedback
                * ap_delay_buffer[delay_index - 1 % len(ap_delay_buffer)]
            )
            ap_delay_buffer[delay_index] = ap_output

            # Store the output
            output[i] = ap_output

        return output

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value, ensuring the parameter is supported and within valid ranges.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or the value is out of the expected range.
        """
        if parameter in ["rate", "depth", "feedback"]:
            if parameter == "rate" and not (0.1 <= value <= 10.0):
                raise ValueError("Rate must be between 0.1 Hz and 10.0 Hz.")
            if parameter == "depth" and not (0.0 <= value <= 1.0):
                raise ValueError("Depth must be between 0.0 and 1.0.")
            if parameter == "feedback" and not (0.0 <= value <= 1.0):
                raise ValueError("Feedback must be between 0.0 and 1.0.")
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class Equalizer(SoundModule):
    """
    Adjusts the balance between frequency components within a sound by applying gain adjustments to specified frequency bands.

    This class inherits from the SoundModule base class and implements the process_sound method to perform audio equalization based on a dictionary of frequency bands and their corresponding gain values.

    Methods:
        process_sound(sound: np.ndarray, frequency_bands: dict) -> np.ndarray:
            Applies equalization to the input sound based on the provided frequency bands.
    """

    def process_sound(self, sound: np.ndarray, frequency_bands: dict) -> np.ndarray:
        """
        Applies equalization to the input sound based on the provided frequency bands. This method adjusts the amplitude of specific frequency ranges according to the gains specified in the frequency_bands dictionary.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            frequency_bands (dict): A dictionary specifying the gain (in dB) for each frequency band. Each key is a tuple representing the frequency range (low_freq, high_freq), and the value is the gain in dB to be applied to that range.

        Returns:
            np.ndarray: The processed sound data with equalization applied.

        Raises:
            ValueError: If the frequency bands are not specified correctly or if the gains are not within a valid range.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Validate the frequency bands and gains
        for freq_range, gain in frequency_bands.items():
            if not isinstance(freq_range, tuple) or not len(freq_range) == 2:
                raise ValueError(
                    "Frequency range must be specified as a tuple (low_freq, high_freq)."
                )
            if not isinstance(gain, (int, float)):
                raise ValueError("Gain must be a numeric value.")
            if not (-60 <= gain <= 12):
                raise ValueError("Gain must be between -60 dB and +12 dB.")

        # Create a copy of the input sound to avoid modifying the original array
        processed_sound = np.copy(sound)

        # Apply the equalization
        for (low_freq, high_freq), gain in frequency_bands.items():
            # Convert frequency range to corresponding indices in the frequency domain
            low_idx = int(low_freq / (SAMPLE_RATE / len(sound)) * 2)
            high_idx = int(high_freq / (SAMPLE_RATE / len(sound)) * 2)

            # Convert gain from dB to linear scale
            linear_gain = 10 ** (gain / 20)

            # Apply the gain to the specified frequency range in the Fourier domain
            sound_fft = np.fft.rfft(processed_sound)
            sound_fft[low_idx:high_idx] *= linear_gain
            processed_sound = np.fft.irfft(sound_fft)

        return processed_sound


class DynamicRangeCompressor(SoundModule):
    """
    Reduces the dynamic range of a sound by applying compression based on a specified threshold and ratio.

    Attributes:
        threshold (float): The threshold level (in dB) above which compression is applied.
        ratio (float): The compression ratio, expressed as input_change:output_change.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies dynamic range compression to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters: 'threshold', 'ratio'.
    """

    def __init__(self, threshold: float, ratio: float) -> None:
        """
        Initializes the DynamicRangeCompressor with the specified threshold and ratio.

        Parameters:
            threshold (float): The threshold level in dB above which the compression starts.
            ratio (float): The ratio of input to output change above the threshold.
        """
        super().__init__()
        self.threshold: float = threshold
        self.ratio: float = ratio

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies dynamic range compression to the input sound based on the threshold and ratio settings.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with dynamic range compression applied.

        Raises:
            ValueError: If the sound array is empty.
        """
        if sound.size == 0:
            raise ValueError("Input sound array cannot be empty.")

        # Convert threshold from dB to linear scale
        threshold_linear = 10 ** (self.threshold / 20)

        # Calculate the gain to be applied based on the threshold and ratio
        gain = np.where(
            sound > threshold_linear,
            threshold_linear + (sound - threshold_linear) / self.ratio,
            sound,
        )

        # Apply the gain to the sound
        processed_sound = sound * gain
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is not within the valid range.
        """
        if parameter in ["threshold", "ratio"]:
            if parameter == "threshold" and not (-60 <= value <= 0):
                raise ValueError("Threshold must be between -60 dB and 0 dB.")
            if parameter == "ratio" and not (1.0 <= value <= 20.0):
                raise ValueError("Ratio must be between 1.0 and 20.0.")
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class DistortionEffect(SoundModule):
    """
    Applies distortion to the sound to achieve a gritty, aggressive tone.

    Methods:
        process_sound(sound: np.ndarray, drive: float, tone: float) -> np.ndarray:
            Applies distortion to the input sound based on drive and tone settings.
    """

    def process_sound(self, sound: np.ndarray, drive: float, tone: float) -> np.ndarray:
        """
        Applies distortion to the input sound based on drive and tone settings.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            drive (float): The amount of distortion to apply (0.0 to 1.0).
            tone (float): The tone of the distortion (-1.0 to 1.0).

        Returns:
            np.ndarray: The processed sound data with distortion applied.
        """
        # TODO: Implement the actual distortion logic
        processed_sound: np.ndarray = sound
        return processed_sound


class StereoPanning(SoundModule):
    """
    Manages the distribution of a sound's signal across a stereo field.

    Methods:
        process_sound(sound: np.ndarray, pan: float) -> np.ndarray:
            Applies stereo panning to the input sound based on the pan parameter.
    """

    def process_sound(self, sound: np.ndarray, pan: float) -> np.ndarray:
        """
        Applies stereo panning to the input sound based on the pan parameter.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            pan (float): The panning position (-1.0 for left, 0.0 for center, 1.0 for right).

        Returns:
            np.ndarray: The processed sound data with stereo panning applied.
        """
        # TODO: Implement the actual stereo panning logic
        processed_sound: np.ndarray = sound
        return processed_sound


class SampleRateAdjustment(SoundModule):
    """
    Adjusts the sample rate of a digital sound signal.

    Methods:
        process_sound(sound: np.ndarray, new_rate: int) -> np.ndarray:
            Resamples the input sound to a new sample rate.
    """

    def process_sound(self, sound: np.ndarray, new_rate: int) -> np.ndarray:
        """
        Resamples the input sound to a new sample rate.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            new_rate (int): The new sample rate to resample the sound to.

        Returns:
            np.ndarray: The processed sound data with the new sample rate.
        """
        # TODO: Implement the actual sample rate adjustment logic
        processed_sound: np.ndarray = sound
        return processed_sound


class BitDepthAdjustment(SoundModule):
    """
    Manages the bit depth of digital audio samples.

    Methods:
        process_sound(sound: np.ndarray, new_depth: int) -> np.ndarray:
            Changes the bit depth of the input sound.
    """

    def process_sound(self, sound: np.ndarray, new_depth: int) -> np.ndarray:
        """
        Changes the bit depth of the input sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            new_depth (int): The new bit depth to apply to the sound.

        Returns:
            np.ndarray: The processed sound data with the new bit depth.
        """
        # TODO: Implement the actual bit depth adjustment logic
        processed_sound: np.ndarray = sound
        return processed_sound


class FormantAdjustment(SoundModule):
    """
    Adjusts the formants in vocal sounds to alter perceived vowel sounds.

    Methods:
        process_sound(sound: np.ndarray, formant_shifts: dict) -> np.ndarray:
            Adjusts formants in the input sound based on the specified shifts.
    """

    def process_sound(self, sound: np.ndarray, formant_shifts: dict) -> np.ndarray:
        """
        Adjusts formants in the input sound based on the specified shifts.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            formant_shifts (dict): A dictionary specifying the shifts for each formant.

        Returns:
            np.ndarray: The processed sound data with formant adjustments applied.
        """
        # TODO: Implement the actual formant adjustment logic
        processed_sound: np.ndarray = sound
        return processed_sound


class NoiseAddition(SoundModule):
    """
    Generates and adds noise to a sound.

    Methods:
        process_sound(sound: np.ndarray, color: str) -> np.ndarray:
            Adds colored noise (e.g., white, pink) to the input sound.
    """

    def process_sound(self, sound: np.ndarray, color: str) -> np.ndarray:
        """
        Adds colored noise (e.g., white, pink) to the input sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            color (str): The color of the noise to add (e.g., 'white', 'pink').

        Returns:
            np.ndarray: The processed sound data with noise added.
        """
        # TODO: Implement the actual noise addition logic
        processed_sound: np.ndarray = sound
        return processed_sound


class TransientShaping(SoundModule):
    """
    Shapes the transients in a sound to modify its attack and decay characteristics.

    Methods:
        process_sound(sound: np.ndarray, attack: float, sustain: float) -> np.ndarray:
            Modifies the attack and sustain characteristics of the input sound.
    """

    def process_sound(
        self, sound: np.ndarray, attack: float, sustain: float
    ) -> np.ndarray:
        """
        Modifies the attack and sustain characteristics of the input sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            attack (float): The attack time in seconds.
            sustain (float): The sustain level (0.0 to 1.0).

        Returns:
            np.ndarray: The processed sound data with modified transients.
        """
        # TODO: Implement the actual transient shaping logic
        processed_sound: np.ndarray = sound
        return processed_sound
