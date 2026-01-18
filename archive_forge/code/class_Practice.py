from suds.sax import Namespace
from suds.sax.element import Element
from suds.plugin import DocumentPlugin, DocumentContext
from logging import getLogger
class Practice(Doctor):
    """
    A collection of doctors.
    @ivar doctors: A list of doctors.
    @type doctors: list
    """

    def __init__(self):
        self.doctors = []

    def add(self, doctor):
        """
        Add a doctor to the practice
        @param doctor: A doctor to add.
        @type doctor: L{Doctor}
        """
        self.doctors.append(doctor)

    def examine(self, root):
        for d in self.doctors:
            d.examine(root)
        return root