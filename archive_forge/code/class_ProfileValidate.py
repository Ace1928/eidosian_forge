from openstack import resource
class ProfileValidate(Profile):
    base_path = '/profiles/validate'
    allow_create = True
    allow_fetch = False
    allow_commit = False
    allow_delete = False
    allow_list = False
    commit_method = 'PUT'