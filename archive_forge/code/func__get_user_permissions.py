from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Exists, OuterRef, Q
def _get_user_permissions(self, user_obj):
    return user_obj.user_permissions.all()