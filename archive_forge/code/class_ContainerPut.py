from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from simpy.core import BoundClass, Environment
from simpy.resources import base
class ContainerPut(base.Put):
    """Request to put *amount* of matter into the *container*. The request will
    be triggered once there is enough space in the *container* available.

    Raise a :exc:`ValueError` if ``amount <= 0``.

    """

    def __init__(self, container: Container, amount: ContainerAmount):
        if amount <= 0:
            raise ValueError(f'amount(={amount}) must be > 0.')
        self.amount = amount
        'The amount of matter to be put into the container.'
        super().__init__(container)