import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
class Nduv:
    """Class representing name-date-unit-value

    Attributes:
        date: date.
        name: name.
        unit: unit.
        value: value.
    """

    def __init__(self, date, name, unit, value):
        """Initialize a new name-date-unit-value object

        Args:
            date (datetime.datetime): Date field
            name (str): Name field
            unit (str): Nduv unit
            value (float): The value of the Nduv
        """
        self.date = date
        self.name = name
        self.unit = unit
        self.value = value

    @classmethod
    def from_dict(cls, data):
        """Create a new Nduv object from a dictionary.

        Args:
            data (dict): A dictionary representing the Nduv to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            Nduv: The Nduv from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the object.

        Returns:
            dict: The dictionary form of the Nduv.
        """
        out_dict = {'date': self.date, 'name': self.name, 'unit': self.unit, 'value': self.value}
        return out_dict

    def __eq__(self, other):
        if isinstance(other, Nduv):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        return f'Nduv({repr(self.date)}, {self.name}, {self.unit}, {self.value})'