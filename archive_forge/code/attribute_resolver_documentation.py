import logging
from saml2 import BINDING_SOAP

        :param name_id: The identifier by which the subject is know
            among all the participents of the VO
        :param issuer: Who am I the poses the query
        :param vo_members: The entity IDs of the IdP who I'm going to ask
            for extra attributes
        :return: A dictionary with all the collected information about the
            subject
        