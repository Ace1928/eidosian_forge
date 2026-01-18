import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def _exercise_project_api(ref_id):
    driver = PROVIDERS.resource_api.driver
    self.assertRaises(exception.ProjectNotFound, driver.get_project, ref_id)
    self.assertRaises(exception.ProjectNotFound, driver.get_project_by_name, resource.NULL_DOMAIN_ID, ref_id)
    project_ids = [x['id'] for x in driver.list_projects(driver_hints.Hints())]
    self.assertNotIn(ref_id, project_ids)
    projects = driver.list_projects_from_ids([ref_id])
    self.assertThat(projects, matchers.HasLength(0))
    project_ids = [x for x in driver.list_project_ids_from_domain_ids([ref_id])]
    self.assertNotIn(ref_id, project_ids)
    self.assertRaises(exception.DomainNotFound, driver.list_projects_in_domain, ref_id)
    project_ids = [x['id'] for x in driver.list_projects_acting_as_domain(driver_hints.Hints())]
    self.assertNotIn(ref_id, project_ids)
    projects = driver.list_projects_in_subtree(ref_id)
    self.assertThat(projects, matchers.HasLength(0))
    self.assertRaises(exception.ProjectNotFound, driver.list_project_parents, ref_id)
    self.assertTrue(driver.is_leaf_project(ref_id))
    self.assertRaises(exception.ProjectNotFound, driver.update_project, ref_id, {})
    self.assertRaises(exception.ProjectNotFound, driver.delete_project, ref_id)
    if ref_id != resource.NULL_DOMAIN_ID:
        driver.delete_projects_from_ids([ref_id])