from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Exists, OuterRef, Q
class AllowAllUsersRemoteUserBackend(RemoteUserBackend):

    def user_can_authenticate(self, user):
        return True