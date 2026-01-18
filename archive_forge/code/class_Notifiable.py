from ansible.playbook.attribute import FieldAttribute
class Notifiable:
    notify = FieldAttribute(isa='list')