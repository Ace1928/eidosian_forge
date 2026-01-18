from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
from simpy.core import BoundClass, Environment
from simpy.resources import base
class ContainerGet(base.Get):
    """Request to get *amount* of matter from the *container*. The request will
    be triggered once there is enough matter available in the *container*.

    Raise a :exc:`ValueError` if ``amount <= 0``.

    """

    def __init__(self, container: Container, amount: ContainerAmount):
        if amount <= 0:
            raise ValueError(f'amount(={amount}) must be > 0.')
        self.amount = amount
        'The amount of matter to be taken out of the container.'
        super().__init__(container)