from novaclient.tests.functional import base
def _create_resize_down_flavors(self):
    """Creates two flavors with different size ram but same size vcpus
        and disk.

        :returns: tuple of 2 IDs which represents larger_flavor for resize and
            smaller flavor.
        """
    output = self.nova('flavor-create', params='%s auto 128 0 1' % self.name_generate())
    larger_id = self._get_column_value_from_single_row_table(output, 'ID')
    self.addCleanup(self.nova, 'flavor-delete', params=larger_id)
    output = self.nova('flavor-create', params='%s auto 64 0 1' % self.name_generate())
    smaller_id = self._get_column_value_from_single_row_table(output, 'ID')
    self.addCleanup(self.nova, 'flavor-delete', params=smaller_id)
    return (larger_id, smaller_id)