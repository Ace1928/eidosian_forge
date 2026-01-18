from __future__ import annotations
import abc
import copy
import os
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile
from monty.io import zopen
from monty.json import MSONable
class InputSet(MSONable, MutableMapping):
    """
    Abstract base class for all InputSet classes. InputSet are dict-like
    containers for all calculation input data.

    Since InputSet inherits dict, it can be instantiated in the same manner,
    or a custom __init__ can be provided. Either way, `self` should be
    populated with keys that are filenames to be written, and values that are
    InputFile objects or strings representing the entire contents of the file.

    All InputSet must implement from_directory. Implementing the validate method
    is optional.
    """

    def __init__(self, inputs: dict[str | Path, str | InputFile] | None=None, **kwargs):
        """
        Instantiate an InputSet.

        Args:
            inputs: The core mapping of filename: file contents that defines the InputSet data.
                This should be a dict where keys are filenames and values are InputFile objects
                or strings representing the entire contents of the file. If a value is not an
                InputFile object nor a str, but has a __str__ method, this str representation
                of the object will be written to the corresponding file. This mapping will
                become the .inputs attribute of the InputSet.
            **kwargs: Any kwargs passed will be set as class attributes e.g.
                InputSet(inputs={}, foo='bar') will make InputSet.foo == 'bar'.
        """
        self.inputs = inputs or {}
        self._kwargs = kwargs
        self.__dict__.update(**kwargs)

    def __getattr__(self, key):
        if key in self._kwargs:
            return self.get(key)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {key!r}")

    def __copy__(self) -> InputSet:
        cls = type(self)
        new_instance = cls.__new__(cls)
        for key, val in self.__dict__.items():
            setattr(new_instance, key, val)
        return new_instance

    def __deepcopy__(self, memo: dict[int, InputSet]) -> InputSet:
        cls = type(self)
        new_instance = cls.__new__(cls)
        memo[id(self)] = new_instance
        for key, val in self.__dict__.items():
            setattr(new_instance, key, copy.deepcopy(val, memo))
        return new_instance

    def __len__(self) -> int:
        return len(self.inputs)

    def __iter__(self) -> Iterator[str | Path]:
        return iter(self.inputs)

    def __getitem__(self, key: str | Path) -> str | InputFile | slice:
        return self.inputs[key]

    def __setitem__(self, key: str | Path, value: str | InputFile) -> None:
        self.inputs[key] = value

    def __delitem__(self, key: str | Path) -> None:
        del self.inputs[key]

    def write_input(self, directory: str | Path, make_dir: bool=True, overwrite: bool=True, zip_inputs: bool=False):
        """
        Write Inputs to one or more files.

        Args:
            directory: Directory to write input files to
            make_dir: Whether to create the directory if it does not already exist.
            overwrite: Whether to overwrite an input file if it already exists.
            Additional kwargs are passed to generate_inputs
            zip_inputs: If True, inputs will be zipped into a file with the
                same name as the InputSet (e.g., InputSet.zip)
        """
        path = directory if isinstance(directory, Path) else Path(directory)
        for fname, contents in self.inputs.items():
            file_path = path / fname
            if not path.exists() and make_dir:
                path.mkdir(parents=True, exist_ok=True)
            if file_path.exists() and (not overwrite):
                raise FileExistsError(fname)
            file_path.touch()
            if isinstance(contents, InputFile):
                contents.write_file(file_path)
            else:
                with zopen(file_path, mode='wt') as file:
                    file.write(str(contents))
        if zip_inputs:
            filename = path / f'{type(self).__name__}.zip'
            with ZipFile(filename, mode='w') as zip_file:
                for fname in self.inputs:
                    file_path = path / fname
                    try:
                        zip_file.write(file_path)
                        os.remove(file_path)
                    except FileNotFoundError:
                        pass

    @classmethod
    def from_directory(cls, directory: str | Path) -> None:
        """
        Construct an InputSet from a directory of one or more files.

        Args:
            directory: Directory to read input files from
        """
        raise NotImplementedError(f'from_directory has not been implemented in {cls.__name__}')

    def validate(self) -> bool:
        """
        A place to implement basic checks to verify the validity of an
        input set. Can be as simple or as complex as desired.

        Will raise a NotImplementedError unless overloaded by the inheriting class.
        """
        raise NotImplementedError(f'.validate() has not been implemented in {type(self).__name__}')