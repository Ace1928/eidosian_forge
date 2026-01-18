import json
class JSONTaggedEncoder(json.JSONEncoder):

    def default(self, obj):
        obj_tag = getattr(obj, 'json_tag', None)
        if obj_tag is None:
            return super().default(obj)
        obj_tag = TAG_PREFIX + obj_tag
        obj = obj.encode_json_obj()
        return {obj_tag: obj}