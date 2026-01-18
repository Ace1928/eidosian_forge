"""
### 3. DAW Core Application
The core of the application manages module integration, user interactions, and real-time sound processing, handling dynamically loaded modules with robust error handling.

"""

import sys
from PyQt5 import QtWidgets
from DAWModules import AmplitudeControl, EnvelopeGenerator
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Modular Sound Synthesizer")
        self.init_modules()
        self.init_ui()
        self.setup_audio_stream()

    def init_modules(self):
        """
        Dynamically loads and initializes sound modules, handling failures gracefully.
        """
        self.modules = {}
        module_classes = [
            AmplitudeControl,
            EnvelopeGenerator,
        ]  # List all module classes
        for module_class in module_classes:
            try:
                module_instance = module_class()
                self.modules[module_class.__name__] = module_instance
            except Exception as e:
                print(f"Failed to load {module_class.__name__}: {e}")

    def init_ui(self):
        """
        Creates UI controls dynamically for each module.
        """
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        for module_name, module in self.modules.items():
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)
            slider.valueChanged.connect(
                lambda value, name=module_name: self.update_module(name, value)
            )
            layout.addWidget(QtWidgets.QLabel(module_name))
            layout.addWidget(slider)
        central_widget.setLayout(layout)

    def update_module(self, module_name: str, value: int):
        """
        Updates module parameters based on GUI controls.
        """
        module = self.modules.get(module_name)
        if module:
            module.set_parameter(
                "volume", value / 100.0
            )  # Example for amplitude control

    def setup_audio_stream(self):
        """
        Sets up real-time audio processing with pyaudio.
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

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Processes audio in real-time.
        """
        data = np.zeros(frame_count, dtype=np.float32)
        for module in self.modules.values():
            data = module.process_sound(data)
        return (data.tobytes(), pyaudio.paContinue)

    def start(self):
        """
        Starts the audio stream.
        """
        self.stream.start_stream()

    def closeEvent(self, event):
        """
        Ensures clean shutdown of the audio stream.
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

"""
### Explanation of the System

- **Module Management**: The application attempts to load all predefined sound modules. If a module fails to initialize (due to missing dependencies or runtime errors), it will catch the exception and continue loading other modules, ensuring robustness.
- **Dynamic GUI**: The GUI automatically generates controls for each loaded module. If a module is not loaded, its controls won't appear, allowing the interface to adapt dynamically to the available functionality.
- **Real-Time Audio Handling**: Audio processing is handled in real-time using PyAudio. Each active module processes the audio stream in succession, applying its effects based on user settings.

This architecture supports high flexibility in terms of module development and integration, ensuring that the DAW can evolve with advancements in sound synthesis and processing technologies. It provides a robust platform for experimentation and production in digital audio.
"""
