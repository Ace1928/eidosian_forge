import logging
def _extract_pkey_values(self, request):
    if request.get('PutRequest'):
        return [request['PutRequest']['Item'][key] for key in self._overwrite_by_pkeys]
    elif request.get('DeleteRequest'):
        return [request['DeleteRequest']['Key'][key] for key in self._overwrite_by_pkeys]
    return None