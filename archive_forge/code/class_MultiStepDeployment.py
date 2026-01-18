import os
import re
import binascii
from typing import IO, List, Union, Optional, cast
from libcloud.utils.py3 import basestring
from libcloud.compute.ssh import BaseSSHClient
from libcloud.compute.base import Node
class MultiStepDeployment(Deployment):
    """
    Runs a chain of Deployment steps.
    """

    def __init__(self, add=None):
        """
        :type add: ``list``
        :keyword add: Deployment steps to add.
        """
        self.steps = []
        if add:
            self.add(add)

    def add(self, add):
        """
        Add a deployment to this chain.

        :type add: Single :class:`Deployment` or a ``list`` of
                   :class:`Deployment`
        :keyword add: Adds this deployment to the others already in this
                      object.
        """
        if add is not None:
            add = add if isinstance(add, (list, tuple)) else [add]
            self.steps.extend(add)

    def run(self, node, client):
        """
        Run each deployment that has been added.

        See also :class:`Deployment.run`
        """
        for s in self.steps:
            node = s.run(node, client)
        return node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        steps = []
        for step in self.steps:
            steps.append(str(step))
        steps = ', '.join(steps)
        return '<MultiStepDeployment steps=[%s]>' % steps