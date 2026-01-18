"""
To design a highly sophisticated Digital Audio Workstation (DAW) that seamlessly integrates all 20 sound synthesis modules, with provisions for dynamic loading and robust error handling, we will create an advanced, modular, and extensible system using Python. This system will leverage the PyQt framework for the GUI, enabling real-time user interaction, and PyAudio for handling audio streams efficiently. Each module will be designed to operate independently, ensuring that the system remains functional even if some modules fail to load.

### 1. Sound Module Base Class
This base class will define a standard interface for all sound processing modules, ensuring uniformity and facilitating easier maintenance and enhancements.
"""

import numpy as np
import scipy.signal
from scipy.signal import resample, fftconvolve
from typing import Union, Tuple, Dict, Any
from abc import ABC, abstractmethod


SAMPLE_RATE = 44100  # Default sample rate for audio processing


class SoundModule(ABC):
    """
    Abstract base class for all sound modules in the Digital Audio Workstation (DAW).
    This class defines the interface and common functionality across all sound modules,
    ensuring a robust, flexible, and efficient foundation for sound processing capabilities.
    """

    def __init__(self) -> None:
        """
        Initialize the SoundModule with default configurations or state setups if necessary.
        This constructor may be extended by subclasses to include specific initializations.
        """
        super().__init__()  # Ensures proper initialization chaining in derived classes

    @abstractmethod
    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Abstract method to process the sound data. This method must be implemented by each module
        to modify the audio signal according to the module's specific sound processing algorithm.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data, which has been modified by the module's algorithm.

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        raise NotImplementedError(
            "Each module must implement the 'process_sound' method."
        )

    @abstractmethod
    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Abstract method to set parameters for the sound module. This method should be implemented
        by modules that have configurable parameters, allowing dynamic adjustment of module behavior.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            NotImplementedError: If the method is not implemented in a derived class.
        """
        raise NotImplementedError(
            "This method should be overridden by modules that have parameters."
        )

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validates the parameters provided to the module to ensure they meet expected criteria.
        This method can be overridden by subclasses to implement module-specific validation logic.

        Parameters:
            parameters (Dict[str, Any]): A dictionary of parameter names and values to be validated.

        Raises:
            ValueError: If any parameter is invalid or out of the expected range.
        """
        for param, value in parameters.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Parameter {param} expects a numeric value, got {type(value).__name__} instead."
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
        super().__init__()  # Proper initialization chaining to the base class SoundModule
        self.volume: float = initial_volume  # Set the initial volume of the module.

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Processes the input sound data by applying amplitude control based on the current volume setting.

        Args:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after applying amplitude control. The amplitude of the sound is
                        adjusted by multiplying the sound data by the current volume level.

        Raises:
            TypeError: If the input sound is not a NumPy array.
        """
        # Validate the input sound type for proper processing.
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
    The EnvelopeGenerator class, inheriting from the SoundModule base class, is meticulously designed to generate an ADSR (Attack, Decay, Sustain, Release) envelope for sound shaping. This class provides a sophisticated mechanism to dynamically shape the amplitude of a sound signal over time according to meticulously defined ADSR envelope parameters.

    Attributes:
        attack (float): The attack time of the envelope in seconds, defining how quickly the sound reaches its peak amplitude.
        decay (float): The decay time of the envelope in seconds, defining how quickly the sound reduces to the sustain level after the initial peak.
        sustain (float): The sustain level of the envelope, representing the amplitude level during the main sequence of the sound's duration, before the release starts.
        release (float): The release time of the envelope in seconds, defining how quickly the sound fades out after the sustain phase.
    """

    def __init__(
        self, attack: float, decay: float, sustain: float, release: float
    ) -> None:
        """
        Initializes the EnvelopeGenerator module with the specified ADSR parameters, ensuring precise control over the sound's amplitude dynamics.

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
        Applies the ADSR envelope to the input sound. This method meticulously modifies the amplitude of the sound data over time according to the ADSR parameters, ensuring a dynamic and expressive sound modulation.

        Args:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after applying the ADSR envelope, reflecting the dynamic changes in amplitude over time.

        Raises:
            TypeError: If the input sound is not a NumPy array.
            ValueError: If the ADSR parameters do not allow for a valid sustain phase with the given sound length.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        num_samples = len(sound)
        attack_samples = int(self.attack * SAMPLE_RATE)
        decay_samples = int(self.decay * SAMPLE_RATE)
        release_samples = int(self.release * SAMPLE_RATE)
        sustain_samples = num_samples - (
            attack_samples + decay_samples + release_samples
        )

        if sustain_samples < 1:
            raise ValueError(
                "ADSR envelope parameters do not allow for a valid sustain phase with the given sound length."
            )

        envelope = np.concatenate(
            [
                np.linspace(0, 1, attack_samples),  # Attack phase
                np.linspace(1, self.sustain, decay_samples),  # Decay phase
                np.full(sustain_samples, self.sustain),  # Sustain phase
                np.linspace(self.sustain, 0, release_samples),  # Release phase
            ]
        )

        processed_sound = sound * envelope
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported, ensuring dynamic configurability of the ADSR envelope parameters.

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
    Manages the pitch alterations of a sound by adjusting its frequency. This class inherits from the
    SoundModule base class and provides specific functionality to modify the pitch of the sound based
    on a given base frequency.

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
        self.validate_parameters({"base_frequency": base_frequency})

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
        self.validate_parameters({"harmonics": harmonics})
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
            self.validate_parameters({"harmonics": value})
            self.harmonics = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class HarmonicGenerator(SoundModule):
    """
    The HarmonicGenerator class, inheriting from the SoundModule base class, is meticulously designed to generate and manipulate overtones above the fundamental frequency. This class provides a sophisticated mechanism to dynamically manipulate the harmonic content of a sound signal based on a fundamental frequency.

    Attributes:
        fundamental_frequency (float): The fundamental frequency of the sound in Hz, serving as the base for harmonic generation.
        overtones (dict): A dictionary where keys are harmonic indices (integers) and values are amplitude multipliers (floats), dictating the strength of each overtone.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies harmonic generation to the input sound by adding overtones based on the fundamental frequency.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value if it is supported, specifically the 'fundamental_frequency'.
    """

    def __init__(self, fundamental_frequency: float) -> None:
        """
        Initializes the HarmonicGenerator module with the specified fundamental frequency, setting up the initial state for harmonic generation.

        Args:
            fundamental_frequency (float): The fundamental frequency in Hz, which is the base for generating harmonics.
        """
        super().__init__()
        self.fundamental_frequency: float = fundamental_frequency
        self.overtones: dict = {
            1: 1.0
        }  # Initialize with the fundamental frequency only, represented as the first harmonic.

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies harmonic generation to the input sound by adding overtones based on the fundamental frequency. This method enhances the harmonic content of the sound by selectively amplifying its frequency components.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with harmonic generation applied, enriching the sound's texture.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Perform a Fourier transform on the input sound to analyze its frequency components
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

        # Perform an inverse Fourier transform to convert the frequency domain back to the time domain
        processed_sound = np.fft.irfft(harmonic_spectrum, n=len(sound))
        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported, specifically for adjusting the 'fundamental_frequency'.

        Parameters:
            parameter (str): The name of the parameter to set.
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
    Implements advanced modulation techniques such as Amplitude Modulation (AM), Frequency Modulation (FM),
    and Phase Modulation (PM) on sound data. This class inherits from the SoundModule base class and
    provides specialized functionality to modulate the sound based on the specified parameters, enhancing
    the auditory experience with depth and variation.

    Attributes:
        modulation_type (str): The type of modulation to apply ('AM', 'FM', or 'PM').
        modulation_frequency (float): The frequency of the modulation signal in Hz.
        modulation_depth (float): The depth of the modulation effect, ranging from 0.0 (no effect) to 1.0 (full effect).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the specified modulation technique to the input sound, modifying its characteristics based on the modulation parameters.
        set_parameter(parameter: str, value: Union[str, float]) -> None:
            Dynamically sets the specified parameter to the given value, supporting real-time audio processing adjustments.
    """

    def __init__(
        self, modulation_type: str, modulation_frequency: float, modulation_depth: float
    ) -> None:
        """
        Initializes the ModulationTechniques module with the specified modulation type, frequency, and depth,
        setting up the necessary configurations for sound modulation.

        Args:
            modulation_type (str): The type of modulation to apply ('AM', 'FM', or 'PM').
            modulation_frequency (float): The frequency of the modulation signal in Hz.
            modulation_depth (float): The depth of the modulation effect, from 0.0 to 1.0.
        """
        super().__init__()
        self.modulation_type: str = modulation_type
        self.modulation_frequency: float = modulation_frequency
        self.modulation_depth: float = modulation_depth
        self.validate_parameters(
            {
                "modulation_type": modulation_type,
                "modulation_frequency": modulation_frequency,
                "modulation_depth": modulation_depth,
            }
        )

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the specified modulation technique to the input sound, utilizing advanced mathematical models
        to achieve the desired modulation effect.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data, which has been modulated according to the specified technique.

        Raises:
            ValueError: If the modulation type is unsupported or if the input sound does not meet the required specifications.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        t = np.arange(len(sound)) / SAMPLE_RATE  # Time array for modulation

        if self.modulation_type == "AM":
            carrier = np.sin(2 * np.pi * self.modulation_frequency * t)
            processed_sound = (1 + self.modulation_depth * carrier) * sound
        elif self.modulation_type == "FM":
            carrier = np.sin(
                2 * np.pi * self.modulation_frequency * t
                + self.modulation_depth
                * np.sin(2 * np.pi * self.modulation_frequency * t)
            )
            processed_sound = sound * carrier
        elif self.modulation_type == "PM":
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
        Sets the specified parameter to the given value, ensuring that the parameter adjustments are dynamically
        applied to the modulation settings.

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
        self.validate_parameters({parameter: value})


class ReverbEffect(SoundModule):
    """
    Simulates reverberation effects mimicking sound reflections in various environments, inheriting from the SoundModule.

    Attributes:
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
        Initializes the ReverbEffect module with specified decay, pre-delay, and mix parameters, ensuring alignment with the SoundModule's initialization protocol.

        Args:
            decay (float): The decay time in seconds, dictating the duration of the reverberation.
            pre_delay (float): The pre-delay time in seconds, setting the initial delay before the reverberation begins.
            mix (float): The mix ratio between the original and reverberated sound, determining the balance of the effect.
        """
        super().__init__()  # Initialize the base class constructor
        self.decay: float = decay
        self.pre_delay: float = pre_delay
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the reverb effect to the input sound using convolution with an exponentially decaying noise signal, adhering to the abstract method contract of the SoundModule.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the reverb effect applied, blending the original and reverberated sounds based on the 'mix' parameter.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

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
        Generates an impulse response using an exponentially decaying noise signal, providing the basis for the reverb effect.

        Parameters:
            length (int): The length of the impulse response, determined by the input sound length.

        Returns:
            np.ndarray: The generated impulse response, an array of exponentially decaying values modulated by random noise.
        """
        t = np.linspace(0, self.decay, length)
        impulse_response = np.exp(-t / self.decay) * np.random.randn(length)
        return impulse_response

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value, ensuring compliance with the abstract method of the SoundModule.

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
    This class implements an echo effect by delaying and replaying the sound. It inherits from the SoundModule
    base class and adheres to its interface, ensuring compatibility and functionality within the Digital Audio
    Workstation (DAW) framework.

    Attributes:
        delay_time (float): The delay time between echoes, measured in seconds.
        feedback (float): The feedback amount, controlling the echo strength, ranging from 0.0 (no feedback) to 1.0 (full feedback).
        mix (float): The mix ratio between the original sound and the echoed sound, ranging from 0.0 (all original) to 1.0 (all echoed).

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

        Raises:
            ValueError: If any parameter is out of the expected range.
        """
        super().__init__()
        self.validate_parameters(
            {"delay_time": delay_time, "feedback": feedback, "mix": mix}
        )
        self.delay_time: float = delay_time
        self.feedback: float = feedback
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the echo effect to the input sound by creating a delayed version of the sound and mixing it back with the original sound.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the echo effect applied, blending the original and echoed sounds based on the 'mix' parameter.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if not isinstance(sound, np.ndarray) or sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

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
        Sets the specified parameter to the given value if it is supported, ensuring compliance with the abstract method of the SoundModule.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or if the value is out of the expected range.
        """
        supported_parameters = {
            "delay_time": (0.0, 10.0),
            "feedback": (0.0, 1.0),
            "mix": (0.0, 1.0),
        }
        if parameter in supported_parameters:
            min_val, max_val = supported_parameters[parameter]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{parameter.capitalize()} must be between {min_val} and {max_val}."
                )
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
        self.validate_parameters({"rate": rate, "depth": depth, "mix": mix})
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
        if not isinstance(sound, np.ndarray) or sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

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
        supported_parameters = {
            "rate": (0.01, 10.0),
            "depth": (0.0, 1.0),
            "mix": (0.0, 1.0),
        }
        if parameter in supported_parameters:
            min_val, max_val = supported_parameters[parameter]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{parameter.capitalize()} must be between {min_val} and {max_val}."
                )
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class FlangerEffect(SoundModule):
    """
    The FlangerEffect class, inheriting from the SoundModule base class, meticulously creates a flanging effect by mixing the sound with a delayed version of itself, modulated over time. This class provides a sophisticated mechanism to dynamically manipulate the delay and feedback of the sound signal based on the modulation rate and depth.

    Attributes:
        delay (float): The base delay time in seconds, dictating the initial delay of the sound.
        depth (float): The depth of the flanging effect, ranging from 0.0 (no effect) to 1.0 (full effect).
        rate (float): The modulation rate of the flanging effect in Hz, controlling the speed of the modulation.
        feedback (float): The feedback amount, controlling the intensity of the flanging effect from 0.0 (no feedback) to 1.0 (maximum feedback).
        mix (float): The mix level between the original and flanged sound, from 0.0 (all original) to 1.0 (all flanged).

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies the flanger effect to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value if it is supported.
            Supported parameters: 'delay', 'depth', 'rate', 'feedback', 'mix'.
    """

    def __init__(
        self, delay: float, depth: float, rate: float, feedback: float, mix: float
    ) -> None:
        """
        Initializes the FlangerEffect module with the specified parameters, setting up the initial state for flanging effect generation.

        Args:
            delay (float): The base delay time in seconds.
            depth (float): The depth of the flanging effect.
            rate (float): The modulation rate of the flanging effect in Hz.
            feedback (float): The feedback amount controlling the flanging strength.
            mix (float): The mix level between the original and flanged sound.
        """
        super().__init__()
        self.delay: float = delay
        self.depth: float = depth
        self.rate: float = rate
        self.feedback: float = feedback
        self.mix: float = mix

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies the flanger effect to the input sound, utilizing a modulated delay line to create the effect.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with the flanger effect applied, blending the original and delayed sounds based on the mix parameter.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

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
        Sets the specified parameter to the given value if it is supported, ensuring the parameter is within valid ranges.

        Parameters:
            parameter (str): The name of the parameter to set.
                             Supported parameters: 'delay', 'depth', 'rate', 'feedback', 'mix'.
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the specified parameter is not supported or the value is out of the expected range.
        """
        supported_parameters = {
            "delay": (0.001, 0.1),
            "depth": (0.0, 1.0),
            "rate": (0.1, 5.0),
            "feedback": (0.0, 1.0),
            "mix": (0.0, 1.0),
        }
        if parameter in supported_parameters:
            min_val, max_val = supported_parameters[parameter]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{parameter.capitalize()} must be between {min_val} and {max_val}."
                )
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class PhaserEffect(SoundModule):
    """
    Implements a phaser effect by modulating the phase of the input sound signal to create peaks and troughs,
    resulting in a sweeping effect that is commonly used in audio processing.

    Attributes:
        rate (float): The modulation rate of the phaser effect in Hz.
        depth (float): The depth of the phaser effect, indicating the extent of phase modulation.
        feedback (float): The feedback amount, which controls the intensity of the effect by feeding the output back into the input.

    Inherits:
        SoundModule: Inherits the abstract base class SoundModule, ensuring compliance with the sound processing interface.
    """

    def __init__(self, rate: float, depth: float, feedback: float) -> None:
        """
        Initializes the PhaserEffect module with specified modulation rate, depth, and feedback parameters.

        Args:
            rate (float): The modulation rate in Hz.
            depth (float): The modulation depth, ranging from 0.0 (no effect) to 1.0 (full effect).
            feedback (float): The feedback amount, ranging from 0.0 (no feedback) to 1.0 (maximum feedback).

        Raises:
            ValueError: If any of the parameters are out of the acceptable range.
        """
        super().__init__()
        self.validate_parameters({"rate": rate, "depth": depth, "feedback": feedback})
        self.rate: float = rate
        self.depth: float = depth
        self.feedback: float = feedback

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Processes the input sound by applying a phaser effect using a series of all-pass filters that modulate the phase.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The sound data after the phaser effect has been applied, creating a sweeping, dynamic change in the sound's phase.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        num_samples = len(sound)
        lfo = np.sin(
            2 * np.pi * np.arange(num_samples) * self.rate / SAMPLE_RATE
        )  # Low Frequency Oscillator for modulation
        output = np.zeros_like(sound)
        ap_delay_buffer = np.zeros(
            int(SAMPLE_RATE / self.rate)
        )  # All-pass filter delay buffer

        for i in range(num_samples):
            modulated_delay = int(
                self.depth * SAMPLE_RATE / self.rate * (1 + lfo[i]) / 2
            )
            delay_index = i - modulated_delay
            if delay_index < 0:
                delay_index += len(ap_delay_buffer)

            ap_output = (
                -self.feedback * ap_delay_buffer[delay_index]
                + sound[i]
                + self.feedback
                * ap_delay_buffer[(delay_index - 1) % len(ap_delay_buffer)]
            )
            ap_delay_buffer[delay_index] = ap_output
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
        supported_parameters = {
            "rate": (0.1, 10.0),
            "depth": (0.0, 1.0),
            "feedback": (0.0, 1.0),
        }
        if parameter in supported_parameters:
            min_val, max_val = supported_parameters[parameter]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{parameter.capitalize()} must be between {min_val} and {max_val}."
                )
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")


class Equalizer(SoundModule):
    """
    The Equalizer class is a sophisticated sound processing module designed to adjust the balance between frequency components within a sound. It achieves this by applying precise gain adjustments to specified frequency bands, thereby enhancing the audio quality and fidelity.

    This class inherits from the SoundModule base class and meticulously implements the process_sound method to perform audio equalization based on a dictionary of frequency bands and their corresponding gain values.

    Attributes:
        frequency_bands (dict): A dictionary specifying the gain adjustments for each frequency band. Each key is a tuple representing the frequency range (low_freq, high_freq), and the value is the gain in dB to be applied to that range.

    Methods:
        process_sound(sound: np.ndarray, frequency_bands: dict) -> np.ndarray:
            Applies meticulous equalization to the input sound based on the provided frequency bands, adjusting the amplitude of specific frequency ranges as specified.
    """

    def __init__(self, frequency_bands: dict) -> None:
        """
        Initializes the Equalizer with specified frequency bands and their corresponding gains.

        Parameters:
            frequency_bands (dict): A dictionary where each key is a tuple representing the frequency range (low_freq, high_freq), and each value is the gain in dB to be applied to that range.
        """
        super().__init__()
        self.validate_parameters(frequency_bands)
        self.frequency_bands = frequency_bands

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Applies equalization to the input sound based on the frequency bands stored in the instance. This method adjusts the amplitude of specific frequency ranges according to the gains specified in the frequency_bands dictionary.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data with equalization meticulously applied.

        Raises:
            ValueError: If the frequency bands are not specified correctly or if the gains are not within a valid range.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Validate the frequency bands and gains
        for freq_range, gain in self.frequency_bands.items():
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
        for (low_freq, high_freq), gain in self.frequency_bands.items():
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

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Dynamically sets a parameter of the Equalizer, adjusting the gain for a specified frequency band.

        Parameters:
            parameter (str): The frequency range as a string tuple ('low_freq, high_freq').
            value (float): The new gain value in dB to be set for the specified frequency range.

        Raises:
            ValueError: If the specified parameter is not a valid frequency range or if the value is not within the valid range of -60 dB to +12 dB.
        """
        if parameter not in self.frequency_bands:
            raise ValueError(f"Unsupported frequency range: {parameter}")
        if not (-60 <= value <= 12):
            raise ValueError("Gain must be between -60 dB and +12 dB.")
        self.frequency_bands[eval(parameter)] = value


class DynamicRangeCompressor(SoundModule):
    """
    Implements dynamic range compression on audio signals. This class inherits from the SoundModule base class
    and provides functionality to reduce the dynamic range of sound by applying compression based on specified
    threshold and ratio parameters.

    Attributes:
        threshold (float): The level (in dB) above which compression is applied.
        ratio (float): The compression ratio, expressed as input_change:output_change.

    Methods:
        process_sound(sound: np.ndarray) -> np.ndarray:
            Applies dynamic range compression to the input sound.
        set_parameter(parameter: str, value: float) -> None:
            Sets the specified parameter to the given value.
            Supported parameters include 'threshold' and 'ratio'.
    """

    def __init__(self, threshold: float, ratio: float) -> None:
        """
        Initializes the DynamicRangeCompressor with the specified threshold and ratio.

        Parameters:
            threshold (float): The threshold level in dB above which the compression starts.
            ratio (float): The ratio of input to output change above the threshold.

        Raises:
            ValueError: If the threshold or ratio parameters are out of the expected range.
        """
        super().__init__()
        self.validate_parameters({"threshold": threshold, "ratio": ratio})
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
            ValueError: If the sound array is empty or not a one-dimensional NumPy array.
        """
        if sound.size == 0:
            raise ValueError("Input sound array cannot be empty.")
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

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

    This class inherits from the SoundModule base class and implements the distortion effect by manipulating the sound waveforms.
    The distortion effect is achieved by applying a non-linear transformation to the sound data, which can create harmonics and alter the original sound's timbre.

    Methods:
        process_sound(sound: np.ndarray, drive: float, tone: float) -> np.ndarray:
            Applies distortion to the input sound based on drive and tone settings.
    """

    def process_sound(self, sound: np.ndarray, drive: float, tone: float) -> np.ndarray:
        """
        Applies distortion to the input sound based on drive and tone settings.

        The distortion effect is implemented by first applying a gain to the sound based on the 'drive' parameter,
        then applying a non-linear function to create the distortion effect, and finally adjusting the tone
        using a low-pass filter.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            drive (float): The amount of distortion to apply (0.0 to 1.0).
            tone (float): The tone of the distortion (-1.0 to 1.0).

        Returns:
            np.ndarray: The processed sound data with distortion applied.

        Raises:
            ValueError: If the input parameters are out of the expected range.
        """
        if not (0.0 <= drive <= 1.0):
            raise ValueError("Drive must be between 0.0 and 1.0.")
        if not (-1.0 <= tone <= 1.0):
            raise ValueError("Tone must be between -1.0 and 1.0.")
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")

        # Apply drive gain
        gain = 1 + drive * 10  # Increasing the gain based on the drive
        driven_sound = sound * gain

        # Apply distortion effect using a tanh function for soft clipping
        distorted_sound = np.tanh(driven_sound)

        # Tone control using a low-pass filter
        # The cutoff frequency is adjusted based on the tone parameter
        cutoff_frequency = 5000 + (
            4500 * (tone + 1) / 2
        )  # Mapping tone to frequency range
        b, a = scipy.signal.butter(2, cutoff_frequency, btype="low", fs=SAMPLE_RATE)
        processed_sound = scipy.signal.filtfilt(b, a, distorted_sound)

        return processed_sound

class StereoPanning(SoundModule):
    """
    This class is responsible for managing the distribution of a sound's signal across a stereo field. It transforms a mono sound input into a stereo output by adjusting the balance between the left and right channels based on the pan parameter. This class inherits from the SoundModule abstract base class and implements the required methods to ensure compliance with the sound processing capabilities defined therein.

    Attributes:
        pan (float): The panning position where -1.0 represents fully left, 0.0 represents center, and 1.0 represents fully right.
    """

    def __init__(self, pan: float = 0.0) -> None:
        """
        Initializes the StereoPanning module with a default or specified pan parameter.

        Parameters:
            pan (float): The initial panning position. Defaults to 0.0 (center).
        """
        super().__init__()
        self.validate_parameters({'pan': pan})
        self.pan: float = pan

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Processes the input sound data by applying stereo panning based on the pan parameter, producing a stereo output.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array, expected to be mono (1D).

        Returns:
            np.ndarray: The processed sound data with stereo panning applied, formatted as a stereo (2D) array.

        Raises:
            ValueError: If the input sound is not a one-dimensional mono array.
            ValueError: If the pan parameter is not within the valid range of -1.0 to 1.0.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional mono array.")
        if not (-1.0 <= self.pan <= 1.0):
            raise ValueError("Pan must be between -1.0 and 1.0.")

        # Calculate left and right channel gains based on the pan parameter
        left_gain = np.cos((self.pan + 1) * np.pi / 4)
        right_gain = np.sin((self.pan + 1) * np.pi / 4)

        # Create a stereo sound array
        stereo_sound = np.zeros((2, len(sound)), dtype=sound.dtype)
        stereo_sound[0, :] = sound * left_gain  # Left channel
        stereo_sound[1, :] = sound * right_gain  # Right channel

        return stereo_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported by this module.

        Parameters:
            parameter (str): The name of the parameter to set ('pan' in this case).
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the parameter is not supported or the value is out of the expected range.
        """
        if parameter == 'pan':
            if not (-1.0 <= value <= 1.0):
                raise ValueError("Pan must be between -1.0 and 1.0.")
            self.pan = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

class SampleRateAdjustment(SoundModule):
    """
    This class, inheriting from the SoundModule base class, meticulously adjusts the sample rate of a digital sound signal.
    It provides high fidelity and precision in the resampling process, ensuring the audio quality is preserved or enhanced.

    Attributes:
        original_rate (int): The original sample rate of the sound before adjustment. This is set to the default sample
                             rate unless specified otherwise.

    Methods:
        process_sound(sound: np.ndarray, new_rate: int) -> np.ndarray:
            Resamples the input sound to a new sample rate, applying high-quality resampling algorithms to preserve
            the audio quality.
    """

    def __init__(self, original_rate: int = SAMPLE_RATE) -> None:
        """
        Initializes the SampleRateAdjustment module with the specified original sample rate.

        Args:
            original_rate (int): The original sample rate of the sound. Defaults to SAMPLE_RATE.
        """
        super().__init__()
        self.original_rate: int = original_rate
        self.validate_parameters({'original_rate': original_rate})

    def process_sound(self, sound: np.ndarray, new_rate: int) -> np.ndarray:
        """
        Resamples the input sound to a new sample rate using high-quality resampling algorithms. This method ensures
        that the resampled sound maintains as much of the original sound's quality as possible.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            new_rate (int): The new sample rate to resample the sound to.

        Returns:
            np.ndarray: The processed sound data with the new sample rate applied.

        Raises:
            ValueError: If the input sound is not a one-dimensional NumPy array.
            ValueError: If the new sample rate is not a positive integer.
        """
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")
        if new_rate <= 0:
            raise ValueError("New sample rate must be a positive integer.")

        # Calculate the resampling factor
        resampling_factor: float = new_rate / self.original_rate

        # Perform resampling using high-quality interpolation
        number_of_samples: int = int(len(sound) * resampling_factor)
        processed_sound: np.ndarray = scipy.signal.resample(sound, number_of_samples)

        return processed_sound

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported by this module.

        Parameters:
            parameter (str): The name of the parameter to set ('original_rate' in this case).
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the parameter is not supported or the value is out of the expected range.
        """
        if parameter == 'original_rate':
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Original rate must be a positive integer.")
            self.original_rate = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")
class BitDepthAdjustment(SoundModule):
    """
    Manages the bit depth of digital audio samples, allowing for the precise adjustment of the resolution
    of audio data. This module inherits from the SoundModule base class and provides functionality to
    modify the bit depth of sound samples, which can affect the dynamic range and noise level of the audio.

    Methods:
        process_sound(sound: np.ndarray, new_depth: int) -> np.ndarray:
            Changes the bit depth of the input sound to the specified new depth.
    """

    def __init__(self) -> None:
        """
        Initializes the BitDepthAdjustment module with default configurations or state setups if necessary.
        This constructor extends the base class constructor to include specific initializations for bit depth adjustment.
        """
        super().__init__()  # Ensures proper initialization chaining in derived classes

    def process_sound(self, sound: np.ndarray, new_depth: int) -> np.ndarray:
        """
        Changes the bit depth of the input sound to the specified new depth. This method performs quantization
        of the audio samples to fit the new bit depth, potentially reducing the data size and altering the
        sound quality by introducing quantization noise.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            new_depth (int): The new bit depth to apply to the sound, must be between 1 and 32.

        Returns:
            np.ndarray: The processed sound data with the new bit depth applied.

        Raises:
            ValueError: If the new_depth is not within the valid range (1 to 32).
            TypeError: If the input sound is not a one-dimensional NumPy array.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("Input sound must be a NumPy array.")
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")
        if not (1 <= new_depth <= 32):
            raise ValueError("New bit depth must be between 1 and 32.")

        # Normalize sound samples to the range of -1 to 1
        max_val = np.max(np.abs(sound))
        normalized_sound = sound / max_val if max_val != 0 else sound

        # Calculate the maximum integer value for the new bit depth
        max_int_value = 2 ** (new_depth - 1) - 1

        # Quantize the normalized sound to the new bit depth
        quantized_sound = np.round(normalized_sound * max_int_value)

        # Scale back to original sound range
        processed_sound = quantized_sound / max_int_value * max_val

        return processed_sound.astype(sound.dtype)

    def set_parameter(self, parameter: str, value: float) -> None:
        """
        Sets the specified parameter to the given value if it is supported by this module.

        Parameters:
            parameter (str): The name of the parameter to set ('new_depth' in this case).
            value (float): The value to set the parameter to.

        Raises:
            ValueError: If the parameter is not supported or the value is out of the expected range.
        """
        if parameter == 'new_depth':
            if not isinstance(value, int) or not (1 <= value <= 32):
                raise ValueError("New depth must be an integer between 1 and 32.")
            self.new_depth = value
        else:
            raise ValueError(f"Unsupported parameter: {parameter}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validates the parameters provided to the module to ensure they meet expected criteria.
        This method can be overridden by subclasses to implement module-specific validation logic.

        Parameters:
            parameters (Dict[str, Any]): A dictionary of parameter names and values to be validated.

        Raises:
            ValueError: If any parameter is invalid or out of the expected range.
        """
        super().validate_parameters(parameters)  # Call to base class method to maintain validation logic
        for param, value in parameters.items():
            if param == "new_depth" and not (1 <= value <= 32):
                raise ValueError(f"New depth must be between 1 and 32, got {value} instead.")

class FormantAdjustment(SoundModule):
    """
    Adjusts the formants in vocal sounds to alter perceived vowel sounds. This class inherits from the
    SoundModule base class and provides functionality to modify the formant frequencies of sound samples,
    which can significantly alter the character and timbre of the audio.

    Methods:
        process_sound(sound: np.ndarray, formant_shifts: dict) -> np.ndarray:
            Adjusts formants in the input sound based on the specified shifts.
    """

    def process_sound(self, sound: np.ndarray, formant_shifts: dict) -> np.ndarray:
        """
        Adjusts formants in the input sound based on the specified shifts. This method applies digital
        signal processing techniques to shift the formant frequencies of the input sound according to
        the provided dictionary mapping formant indices to their respective shifts in Hertz.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            formant_shifts (dict): A dictionary specifying the shifts for each formant in Hertz.

        Returns:
            np.ndarray: The processed sound data with formant adjustments applied.

        Raises:
            ValueError: If the formant_shifts dictionary contains non-numeric values.
            TypeError: If the input sound is not a one-dimensional NumPy array.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("Input sound must be a NumPy array.")
        if sound.ndim != 1:
            raise ValueError("Input sound must be a one-dimensional NumPy array.")
        if not all(
            isinstance(shift, (int, float)) for shift in formant_shifts.values()
        ):
            raise ValueError("All formant shifts must be numeric values.")

        # Perform formant analysis on the input sound
        sample_rate = SAMPLE_RATE
        sound_length = len(sound)
        time_array = np.linspace(0, sound_length / sample_rate, sound_length)
        formant_frequencies, bandwidths = self._analyze_formants(sound, sample_rate)

        # Adjust formant frequencies based on the specified shifts
        adjusted_formants = {
            formant: freq + formant_shifts.get(formant, 0)
            for formant, freq in formant_frequencies.items()
        }

        # Synthesize sound with adjusted formants
        processed_sound = self._synthesize_formants(
            sound, adjusted_formants, bandwidths, sample_rate
        )

        return processed_sound

    def _analyze_formants(
        self, sound: np.ndarray, sample_rate: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Analyzes the formant frequencies and bandwidths of the input sound using LPC (Linear Predictive Coding).

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            sample_rate (int): The sample rate of the sound data.

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]: A tuple containing two dictionaries,
            one for formant frequencies and one for bandwidths, both indexed by formant number.
        """
        # Placeholder for actual LPC analysis implementation
        # This should be replaced with a real LPC analysis method
        return {1: 500, 2: 1500, 3: 2500}, {1: 50, 2: 70, 3: 100}

    def _synthesize_formants(
        self,
        sound: np.ndarray,
        formants: Dict[int, float],
        bandwidths: Dict[int, float],
        sample_rate: int,
    ) -> np.ndarray:
        """
        Synthesizes the sound with adjusted formants using the overlap-add method and frequency domain manipulation.

        Parameters:
            sound (np.ndarray): The original sound data.
            formants (Dict[int, float]): The adjusted formant frequencies.
            bandwidths (Dict[int, float]): The bandwidths of the formants.
            sample_rate (int): The sample rate of the sound data.

        Returns:
            np.ndarray: The sound data with adjusted formants.
        """
        # Placeholder for actual synthesis implementation
        # This should be replaced with a real synthesis method
        return sound  # This line is a placeholder


class NoiseAddition(SoundModule):
    """
    Generates and adds noise to a sound. This class inherits from the SoundModule base class and
    provides functionality to add colored noise (e.g., white, pink) to the input sound data.

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

        Raises:
            ValueError: If the specified color of noise is not supported.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")

        # Validate the color input and generate the corresponding noise
        if color.lower() not in ["white", "pink"]:
            raise ValueError(
                f"Unsupported noise color: {color}. Supported colors are 'white' and 'pink'."
            )

        # Generate noise based on the color
        if color.lower() == "white":
            noise = np.random.normal(0, 1, sound.shape)
        elif color.lower() == "pink":
            # Generating pink noise using Voss-McCartney algorithm
            num_samples = len(sound)
            num_columns = 16
            array = np.random.randn(num_columns, num_samples)
            reshaped_array = np.cumsum(array, axis=1)
            noise = reshaped_array[num_columns - 1] / np.sqrt(num_columns)

        # Normalize noise to match the sound's amplitude
        max_sound = np.max(np.abs(sound))
        max_noise = np.max(np.abs(noise))
        normalized_noise = noise * (max_sound / max_noise)

        # Add the normalized noise to the original sound
        processed_sound = sound + normalized_noise

        return processed_sound.astype(sound.dtype)


class TransientShaping(SoundModule):
    """
    Shapes the transients in a sound to modify its attack and decay characteristics, specifically focusing on the
    manipulation of the initial attack phase and the sustain level of the sound waveform. This module is essential
    for dynamic sound design, allowing for precise control over the sharpness and duration of sound peaks.

    Methods:
        process_sound(sound: np.ndarray, attack: float, sustain: float) -> np.ndarray:
            Modifies the attack and sustain characteristics of the input sound, applying an envelope to enhance
            or suppress transients.
    """

    def process_sound(
        self, sound: np.ndarray, attack: float, sustain: float
    ) -> np.ndarray:
        """
        Modifies the attack and sustain characteristics of the input sound by applying an envelope that enhances
        or suppresses the transients based on the provided attack and sustain parameters.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.
            attack (float): The attack time in seconds, defining how quickly the sound reaches its peak amplitude.
            sustain (float): The sustain level (0.0 to 1.0), representing the amplitude level during the main sequence
                             of the sound's duration, before the release starts.

        Returns:
            np.ndarray: The processed sound data with modified transients, where the attack phase has been
                        dynamically altered and the sustain level adjusted.

        Raises:
            ValueError: If the attack or sustain parameters are out of the expected range.
        """
        if not isinstance(sound, np.ndarray):
            raise TypeError("The input sound must be a NumPy array.")
        if not (0.0 <= attack <= 10.0):
            raise ValueError("Attack time must be between 0.0 and 10.0 seconds.")
        if not (0.0 <= sustain <= 1.0):
            raise ValueError("Sustain level must be between 0.0 and 1.0.")

        # Calculate the number of samples corresponding to the attack time
        attack_samples = int(attack * SAMPLE_RATE)
        if attack_samples > len(sound):
            raise ValueError("Attack time exceeds sound duration.")

        # Apply an attack envelope
        attack_envelope = np.linspace(0, 1, attack_samples)
        sustain_envelope = np.full(len(sound) - attack_samples, sustain)

        # Combine envelopes
        full_envelope = np.concatenate((attack_envelope, sustain_envelope))

        # Apply the envelope to the sound
        processed_sound = sound * full_envelope

        return processed_sound.astype(sound.dtype)
