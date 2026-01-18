from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class ConfigurationTemplate(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(tags=params.get('tags'), author=params.get('author'), composite=params.get('composite'), containingTemplates=params.get('containingTemplates'), createTime=params.get('createTime'), customParamsOrder=params.get('customParamsOrder'), description=params.get('description'), deviceTypes=params.get('deviceTypes'), failurePolicy=params.get('failurePolicy'), id=params.get('id'), language=params.get('language'), lastUpdateTime=params.get('lastUpdateTime'), latestVersionTime=params.get('latestVersionTime'), name=params.get('name'), parentTemplateId=params.get('parentTemplateId'), projectId=params.get('projectId'), projectName=params.get('projectName'), rollbackTemplateContent=params.get('rollbackTemplateContent'), rollbackTemplateParams=params.get('rollbackTemplateParams'), softwareType=params.get('softwareType'), softwareVariant=params.get('softwareVariant'), softwareVersion=params.get('softwareVersion'), templateContent=params.get('templateContent'), templateParams=params.get('templateParams'), validationErrors=params.get('validationErrors'), version=params.get('version'), template_id=params.get('templateId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['project_id'] = self.new_object.get('projectId') or self.new_object.get('project_id')
        new_object_params['software_type'] = self.new_object.get('softwareType') or self.new_object.get('software_type')
        new_object_params['software_version'] = self.new_object.get('softwareVersion') or self.new_object.get('software_version')
        new_object_params['product_family'] = self.new_object.get('productFamily') or self.new_object.get('product_family')
        new_object_params['product_series'] = self.new_object.get('productSeries') or self.new_object.get('product_series')
        new_object_params['product_type'] = self.new_object.get('productType') or self.new_object.get('product_type')
        new_object_params['filter_conflicting_templates'] = self.new_object.get('filterConflictingTemplates') or self.new_object.get('filter_conflicting_templates')
        new_object_params['tags'] = self.new_object.get('tags')
        new_object_params['project_names'] = self.new_object.get('projectName') or self.new_object.get('projectNames') or self.new_object.get('project_names')
        new_object_params['un_committed'] = self.new_object.get('unCommitted') or self.new_object.get('un_committed')
        new_object_params['sort_order'] = self.new_object.get('sortOrder') or self.new_object.get('sort_order')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['template_id'] = self.new_object.get('template_id')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        new_object_params['tags'] = self.new_object.get('tags')
        new_object_params['author'] = self.new_object.get('author')
        new_object_params['composite'] = self.new_object.get('composite')
        new_object_params['containingTemplates'] = self.new_object.get('containingTemplates')
        new_object_params['createTime'] = self.new_object.get('createTime')
        new_object_params['customParamsOrder'] = self.new_object.get('customParamsOrder')
        new_object_params['description'] = self.new_object.get('description')
        new_object_params['deviceTypes'] = self.new_object.get('deviceTypes')
        new_object_params['failurePolicy'] = self.new_object.get('failurePolicy')
        new_object_params['id'] = self.new_object.get('id')
        new_object_params['language'] = self.new_object.get('language')
        new_object_params['lastUpdateTime'] = self.new_object.get('lastUpdateTime')
        new_object_params['latestVersionTime'] = self.new_object.get('latestVersionTime')
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['parentTemplateId'] = self.new_object.get('parentTemplateId')
        new_object_params['projectId'] = self.new_object.get('projectId')
        new_object_params['projectName'] = self.new_object.get('projectName')
        new_object_params['rollbackTemplateContent'] = self.new_object.get('rollbackTemplateContent')
        new_object_params['rollbackTemplateParams'] = self.new_object.get('rollbackTemplateParams')
        new_object_params['softwareType'] = self.new_object.get('softwareType')
        new_object_params['softwareVariant'] = self.new_object.get('softwareVariant')
        new_object_params['softwareVersion'] = self.new_object.get('softwareVersion')
        new_object_params['templateContent'] = self.new_object.get('templateContent')
        new_object_params['templateParams'] = self.new_object.get('templateParams')
        new_object_params['validationErrors'] = self.new_object.get('validationErrors')
        new_object_params['version'] = self.new_object.get('version')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='configuration_templates', function='gets_the_templates_available', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='configuration_templates', function='get_template_details', params={'template_id': id})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('template_id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('templateId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(template_id=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('tags', 'tags'), ('author', 'author'), ('composite', 'composite'), ('containingTemplates', 'containingTemplates'), ('createTime', 'createTime'), ('customParamsOrder', 'customParamsOrder'), ('description', 'description'), ('deviceTypes', 'deviceTypes'), ('failurePolicy', 'failurePolicy'), ('id', 'id'), ('language', 'language'), ('lastUpdateTime', 'lastUpdateTime'), ('latestVersionTime', 'latestVersionTime'), ('name', 'name'), ('parentTemplateId', 'parentTemplateId'), ('projectId', 'projectId'), ('projectName', 'projectName'), ('rollbackTemplateContent', 'rollbackTemplateContent'), ('rollbackTemplateParams', 'rollbackTemplateParams'), ('softwareType', 'softwareType'), ('softwareVariant', 'softwareVariant'), ('softwareVersion', 'softwareVersion'), ('templateContent', 'templateContent'), ('templateParams', 'templateParams'), ('validationErrors', 'validationErrors'), ('version', 'version'), ('templateId', 'template_id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='configuration_templates', function='update_template', params=self.update_all_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('template_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('templateId')
            if id_:
                self.new_object.update(dict(template_id=id_))
        result = self.dnac.exec(family='configuration_templates', function='deletes_the_template', params=self.delete_by_id_params())
        return result