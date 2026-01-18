from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Exists, OuterRef, Q
def _get_permissions(self, user_obj, obj, from_name):
    """
        Return the permissions of `user_obj` from `from_name`. `from_name` can
        be either "group" or "user" to return permissions from
        `_get_group_permissions` or `_get_user_permissions` respectively.
        """
    if not user_obj.is_active or user_obj.is_anonymous or obj is not None:
        return set()
    perm_cache_name = '_%s_perm_cache' % from_name
    if not hasattr(user_obj, perm_cache_name):
        if user_obj.is_superuser:
            perms = Permission.objects.all()
        else:
            perms = getattr(self, '_get_%s_permissions' % from_name)(user_obj)
        perms = perms.values_list('content_type__app_label', 'codename').order_by()
        setattr(user_obj, perm_cache_name, {'%s.%s' % (ct, name) for ct, name in perms})
    return getattr(user_obj, perm_cache_name)