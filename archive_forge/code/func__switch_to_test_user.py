import sys
from django.conf import settings
from django.db import DatabaseError
from django.db.backends.base.creation import BaseDatabaseCreation
from django.utils.crypto import get_random_string
from django.utils.functional import cached_property
def _switch_to_test_user(self, parameters):
    """
        Switch to the user that's used for creating the test database.

        Oracle doesn't have the concept of separate databases under the same
        user, so a separate user is used; see _create_test_db(). The main user
        is also needed for cleanup when testing is completed, so save its
        credentials in the SAVED_USER/SAVED_PASSWORD key in the settings dict.
        """
    real_settings = settings.DATABASES[self.connection.alias]
    real_settings['SAVED_USER'] = self.connection.settings_dict['SAVED_USER'] = self.connection.settings_dict['USER']
    real_settings['SAVED_PASSWORD'] = self.connection.settings_dict['SAVED_PASSWORD'] = self.connection.settings_dict['PASSWORD']
    real_test_settings = real_settings['TEST']
    test_settings = self.connection.settings_dict['TEST']
    real_test_settings['USER'] = real_settings['USER'] = test_settings['USER'] = self.connection.settings_dict['USER'] = parameters['user']
    real_settings['PASSWORD'] = self.connection.settings_dict['PASSWORD'] = parameters['password']