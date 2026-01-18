from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Generic, TypeVar
 Return a list of ``PropertyDescriptor`` instances to install on a
        class, in order to delegate attribute access to this property.

        Args:
            name (str) : the name of the property these descriptors are for

        Returns:
            list[PropertyDescriptor]

        The descriptors returned are collected by the ``MetaHasProps``
        metaclass and added to ``HasProps`` subclasses during class creation.

        Subclasses of ``PropertyDescriptorFactory`` are responsible for
        implementing this function to return descriptors specific to their
        needs.

        