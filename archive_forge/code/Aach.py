"""
### 3. DAW Core Application
The core of the application manages module integration, user interactions, and real-time sound processing, handling dynamically loaded modules with robust error handling.

"""

import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List


class ModuleRegistry:
    """
    A registry for all sound modules in the Digital Audio Workstation (DAW).
    This class provides a centralized listing of all available sound processing modules,
    facilitating dynamic loading, discovery, and robust error handling of modules within the system.
    """

    def __init__(self) -> None:
        """
        Initializes the ModuleRegistry with an empty dictionary to store module references.
        """
        self.modules: Dict[str, Type[SoundModule]] = {}

    def register_module(self, module_name: str, module_ref: Type[SoundModule]) -> None:
        """
        Registers a new sound module in the registry.

        Parameters:
            module_name (str): The name of the module to register.
            module_ref (Type[SoundModule]): A reference to the module class.

        Raises:
            ValueError: If a module with the same name is already registered.
        """
        if module_name in self.modules:
            raise ValueError(f"Module {module_name} is already registered.")
        self.modules[module_name] = module_ref

    def get_module(self, module_name: str) -> Type[SoundModule]:
        """
        Retrieves a module from the registry by its name.

        Parameters:
            module_name (str): The name of the module to retrieve.

        Returns:
            Type[SoundModule]: The module class reference.

        Raises:
            KeyError: If no module with the given name is found.
        """
        if module_name not in self.modules:
            raise KeyError(f"Module {module_name} not found.")
        return self.modules[module_name]

    def list_modules(self) -> List[str]:
        """
        Lists all registered modules' names.

        Returns:
            List[str]: A list of all registered module names.
        """
        return list(self.modules.keys())

    def load_all_modules(self) -> None:
        """
        Dynamically loads all modules from the DAWModules.py file, registering each one.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the module does not conform to the expected interface.
        """
        import os
        import importlib

        module_dir = os.path.dirname(__file__)
        module_files = [
            f
            for f in os.listdir(module_dir)
            if f.endswith(".py") and f == "DAWModules.py"
        ]

        for module_file in module_files:
            module_path = os.path.splitext(module_file)[0]
            module = importlib.import_module(module_path)

            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (
                    isinstance(attribute, type)
                    and issubclass(attribute, SoundModule)
                    and attribute is not SoundModule
                ):
                    try:
                        self.register_module(attribute_name, attribute)
                    except ValueError as e:
                        print(f"Skipping registration for {attribute_name}: {e}")


# Instantiate the module registry
module_registry = ModuleRegistry()

# Dynamically load and register all available modules
module_registry.load_all_modules()


class DAWApplication(QtWidgets.QMainWindow):
    """
    The DAWApplication class extends QtWidgets.QMainWindow to provide a comprehensive user interface for an Advanced Modular Sound Synthesizer. This class is responsible for initializing the application, setting up the user interface, managing sound modules, and handling real-time audio processing.

    Attributes:
        modules (Dict[str, SoundModule]): A dictionary mapping module names to their instances.
        pyaudio_instance (pyaudio.PyAudio): An instance of PyAudio for managing audio streams.
        stream (pyaudio.Stream): The active PyAudio stream for audio output.
    """

    def __init__(self) -> None:
        """
        Initializes the DAWApplication by setting up the window, modules, user interface, and audio stream.
        """
        super().__init__()
        self.setWindowTitle("Advanced Modular Sound Synthesizer")
        self.modules = self.init_modules()
        self.init_ui()
        self.setup_audio_stream()

    def init_modules(self) -> Dict[str, SoundModule]:
        """
        Dynamically loads and initializes sound modules using the ModuleRegistry, handling failures gracefully and ensuring all modules are loaded if possible.

        Returns:
            Dict[str, SoundModule]: A dictionary of module names to their initialized instances.
        """
        module_registry = ModuleRegistry()
        module_registry.load_all_modules()
        modules = {}
        for module_name in module_registry.list_modules():
            module_class = module_registry.get_module(module_name)
            try:
                module_instance = module_class()
                modules[module_name] = module_instance
            except Exception as e:
                print(f"Failed to initialize module {module_name}: {e}")
        return modules

    def init_ui(self) -> None:
        """
        Creates UI controls dynamically for each loaded module, ensuring that each module can be interacted with through the GUI.
        """
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        for module_name, module in self.modules.items():
            label = QtWidgets.QLabel(module_name)
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            slider.valueChanged.connect(
                lambda value, name=module_name: self.update_module(name, value)
            )
            layout.addWidget(label)
            layout.addWidget(slider)

    def update_module(self, module_name: str, value: int) -> None:
        """
        Updates module parameters based on GUI controls, specifically adjusting the 'volume' parameter as an example.

        Parameters:
            module_name (str): The name of the module to update.
            value (int): The new value from the slider, scaled to a range suitable for the module.
        """
        module = self.modules.get(module_name)
        if module:
            scaled_value = value / 100.0
            module.set_parameter("volume", scaled_value)

    def setup_audio_stream(self) -> None:
        """
        Sets up real-time audio processing with PyAudio, configuring the stream for stereo output at 44100 Hz.
        """
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=44100,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback,
        )

    def audio_callback(
        self, in_data, frame_count, time_info, status
    ) -> Tuple[bytes, int]:
        """
        Processes audio in real-time by passing it through each active module sequentially.

        Parameters:
            in_data (bytes): Input audio data (not used in this callback).
            frame_count (int): The number of frames to process.
            time_info (dict): Timing information.
            status (int): Stream status.

        Returns:
            Tuple[bytes, int]: A tuple containing the processed audio data and a flag indicating whether to continue the stream.
        """
        data = np.zeros(frame_count, dtype=np.float32)
        for module in self.modules.values():
            data = module.process_sound(data)
        return (data.tobytes(), pyaudio.paContinue)

    def start(self) -> None:
        """
        Starts the audio stream, beginning real-time audio processing.
        """
        self.stream.start_stream()

    def closeEvent(self, event) -> None:
        """
        Ensures clean shutdown of the audio stream and PyAudio instance upon closing the application.

        Parameters:
            event (QCloseEvent): The close event.
        """
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.pyaudio_instance.terminate()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    daw = DAWApplication()
    daw.show()
    daw.start()
    sys.exit(app.exec_())
