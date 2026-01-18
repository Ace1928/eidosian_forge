import sys
import dns._features
def _config_domain(domain):
    if domain.startswith('.'):
        domain = domain[1:]
    return dns.name.from_text(domain)