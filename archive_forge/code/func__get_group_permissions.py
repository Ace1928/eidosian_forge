from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.db.models import Exists, OuterRef, Q
def _get_group_permissions(self, user_obj):
    user_groups_field = get_user_model()._meta.get_field('groups')
    user_groups_query = 'group__%s' % user_groups_field.related_query_name()
    return Permission.objects.filter(**{user_groups_query: user_obj})