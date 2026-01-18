import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple
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
            raise ValueError(f'Module {module_name} is already registered.')
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
            raise KeyError(f'Module {module_name} not found.')
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
        module_files = [f for f in os.listdir(module_dir) if f.endswith('.py') and f == 'DAWModules.py']
        for module_file in module_files:
            module_path = os.path.splitext(module_file)[0]
            module = importlib.import_module(module_path)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, type) and issubclass(attribute, SoundModule) and (attribute is not SoundModule):
                    try:
                        self.register_module(attribute_name, attribute)
                    except ValueError as e:
                        print(f'Skipping registration for {attribute_name}: {e}')