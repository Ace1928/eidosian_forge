from operator import itemgetter
from breezy import controldir
from ... import errors, osutils, transport
from ...trace import note, show_error
from .helpers import best_format_for_objects_in_a_repository, single_plural
Update a branch with last revision and tag information.

        :return: whether the branch was changed or not
        