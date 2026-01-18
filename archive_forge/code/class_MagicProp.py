import pytest
from nibabel.onetime import auto_attr, setattr_on_read
from nibabel.testing import deprecated_to, expires
class MagicProp:

    @setattr_on_read
    def a(self):
        return object()